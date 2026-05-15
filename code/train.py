"""
Training loop with patient-grouped CV.

Design choices:
- BCEWithLogitsLoss with pos_weight to handle class imbalance.
- AdamW + cosine LR schedule with linear warmup.
- Mixup augmentation (alpha=0.2) for regularisation.
- Label smoothing (eps=0.05) to reduce overconfident predictions.
- Gradient clipping (max_norm=1.0) to stabilise EfficientNet training.
- Early stopping on val AUC (patience=6 epochs).
- DataParallel when multiple GPUs available.
- AUC tracked per epoch; best-AUC checkpoint kept.

Run from Kaggle:
    !python /kaggle/working/code/train.py \
        --data-root /kaggle/input/multimodal-cancer-classification-challenge-2026 \
        --out-dir   /kaggle/working/runs \
        --fold 0 --n-splits 3 --epochs 15 --batch-size 128 \
        --backbone efficientnet_b0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import CellDataset, load_train_df
from model import MultimodalClassifier
from splits import (
    leave_one_patient_out, patient_group_kfold,
    stratified_patient_kfold, summarize_split,
)
from transforms import (
    PairedGeoAug, eval_modality_transform, train_modality_transform,
)


def make_loaders(df, train_idx, val_idx, data_root: Path, batch_size: int, num_workers: int):
    """Build train and val DataLoaders for one CV fold.

    Train loader uses photometric + geometric augmentation, shuffle=True,
    and drop_last=True so BatchNorm always sees a full batch. Val loader
    uses the deterministic eval transform, no shuffle, and drop_last=False
    so every val sample is scored exactly once.
    """
    bf_dir = data_root / "BF" / "train"
    fl_dir = data_root / "FL" / "train"

    train_ds = CellDataset(
        df.iloc[train_idx], bf_dir, fl_dir,
        bf_transform=train_modality_transform("bf"),
        fl_transform=train_modality_transform("fl"),
        paired_transform=PairedGeoAug(),
    )
    val_ds = CellDataset(
        df.iloc[val_idx], bf_dir, fl_dir,
        bf_transform=eval_modality_transform("bf"),
        fl_transform=eval_modality_transform("fl"),
        paired_transform=None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def mixup(bf, fl, y, alpha: float = 0.2):
    """In-place mixup on a batch. Returns mixed (bf, fl, y)."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(bf.size(0), device=bf.device)
    return (lam * bf + (1 - lam) * bf[idx],
            lam * fl + (1 - lam) * fl[idx],
            lam * y  + (1 - lam) * y[idx])


def smooth(y, eps: float = 0.05):
    """Label-smoothing for binary targets: 0/1 -> eps/2 / (1 - eps/2).

    Reduces over-confident predictions and tends to help calibration. With
    eps=0.05, target 1 becomes 0.975 and target 0 becomes 0.025.
    """
    return y * (1.0 - eps) + eps * 0.5


def run_epoch(model, loader, optimizer, scaler, criterion, device, train: bool,
              mixup_alpha: float = 0.0, label_smooth: float = 0.0,
              grad_clip: float = 0.0, sched=None):
    """One pass over `loader`. Returns (mean_loss, AUC, hard_labels, sigmoids).

    train=True does backward + optimizer step; train=False is eval-mode
    inference. AUC is always computed against the un-mixed, un-smoothed
    hard labels so it stays comparable across augmentation settings.
    """
    model.train(train)
    losses, hard_ys, ps = [], [], []
    for batch in loader:
        bf = batch["bf"].to(device, non_blocking=True)
        fl = batch["fl"].to(device, non_blocking=True)
        y  = batch["label"].float().to(device, non_blocking=True)
        hard_ys.append(batch["label"].numpy())  # keep un-mixed labels for AUC

        if train:
            if mixup_alpha > 0:
                bf, fl, y = mixup(bf, fl, y, mixup_alpha)
            if label_smooth > 0:
                y = smooth(y, label_smooth)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(bf, fl)
            loss = criterion(logits, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if sched is not None:
                sched.step()  # OneCycleLR steps per batch

        losses.append(loss.item())
        ps.append(torch.sigmoid(logits).detach().float().cpu().numpy())

    hard_ys = np.concatenate(hard_ys)
    ps = np.concatenate(ps)
    auc = roc_auc_score(hard_ys, ps) if len(np.unique(hard_ys)) > 1 else float("nan")
    return float(np.mean(losses)), auc, hard_ys, ps


def main():
    """Train one fold to completion.

    Reads train.csv, builds the CV splits, trains one fold with mixup +
    label smoothing + OneCycleLR, tracks val AUC, saves the best-AUC
    checkpoint plus an OOF predictions CSV, writes the training history to
    JSON. Early-stops if val AUC fails to improve for `patience` epochs.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--cv", choices=["sgkf", "gkf", "lopo"], default="sgkf")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=6,
                    help="Early-stopping patience (epochs without val-AUC improvement)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--backbone", default="efficientnet_b0",
                    choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_train_df(args.data_root / "train.csv")
    print(f"Loaded {len(df)} rows, {df['patient_id'].nunique()} patients, "
          f"overall pos-rate {df['Diagnosis'].mean():.3f}")

    if args.cv == "sgkf":
        splits = list(stratified_patient_kfold(df, n_splits=args.n_splits, seed=args.seed))
    elif args.cv == "gkf":
        splits = list(patient_group_kfold(df, n_splits=args.n_splits, seed=args.seed))
    elif args.cv == "lopo":
        splits = list(leave_one_patient_out(df))
    print(f"Total folds in this CV: {len(splits)} ({args.cv})")
    train_idx, val_idx = splits[args.fold]
    print(summarize_split(df, train_idx, val_idx))

    train_loader, val_loader = make_loaders(
        df, train_idx, val_idx, args.data_root, args.batch_size, args.num_workers,
    )

    model = MultimodalClassifier(pretrained=not args.no_pretrained,
                                 dropout=args.dropout,
                                 backbone=args.backbone).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    pos = float((df.iloc[train_idx]["Diagnosis"] == 1).sum())
    neg = float((df.iloc[train_idx]["Diagnosis"] == 0).sum())
    pos_weight = torch.tensor(neg / max(pos, 1.0), device=device)
    print(f"pos_weight = {pos_weight.item():.3f}  backbone = {args.backbone}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # OneCycleLR: linear warmup (10%) + cosine decay, stepped per batch.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs,
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    history = []
    best_auc = -1.0
    no_improve = 0
    ckpt_path = args.out_dir / f"fold{args.fold}_best.pt"

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_auc, _, _ = run_epoch(
            model, train_loader, optimizer, scaler, criterion, device, train=True,
            mixup_alpha=args.mixup_alpha, label_smooth=args.label_smooth,
            grad_clip=args.grad_clip, sched=sched,
        )
        with torch.no_grad():
            va_loss, va_auc, va_y, va_p = run_epoch(
                model, val_loader, None, None, criterion, device, train=False,
            )
        dt = time.time() - t0
        print(f"epoch {epoch:>2d} | tr_loss {tr_loss:.4f} tr_auc {tr_auc:.4f} "
              f"| va_loss {va_loss:.4f} va_auc {va_auc:.4f} | {dt:.1f}s")
        history.append({
            "epoch": epoch, "tr_loss": tr_loss, "tr_auc": tr_auc,
            "va_loss": va_loss, "va_auc": va_auc, "time": dt,
        })

        if va_auc > best_auc:
            best_auc = va_auc
            no_improve = 0
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({"model": state, "epoch": epoch, "val_auc": va_auc,
                        "args": vars(args)}, ckpt_path)
            oof = pd.DataFrame({
                "Name": df.iloc[val_idx]["Name"].values,
                "patient_id": df.iloc[val_idx]["patient_id"].values,
                "y_true": va_y,
                "y_pred": va_p,
            })
            oof.to_csv(args.out_dir / f"fold{args.fold}_oof.csv", index=False)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    with open(args.out_dir / f"fold{args.fold}_history.json", "w") as f:
        json.dump({"history": history, "best_auc": best_auc,
                   "args": {k: str(v) for k, v in vars(args).items()}}, f, indent=2)
    print(f"DONE fold {args.fold}, best val AUC = {best_auc:.4f}, ckpt {ckpt_path}")


if __name__ == "__main__":
    main()
