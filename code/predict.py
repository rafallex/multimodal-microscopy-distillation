"""
Inference + submission CSV writer.

Loads one or more fold checkpoints, averages sigmoid scores over folds and
over TTA flips, writes /kaggle/working/submission.csv in the format:

    Name, Diagnosis
    image_1.jpg, 0.3
    ...

Run from Kaggle:
    !python /kaggle/working/code/predict.py \
        --data-root /kaggle/input/multimodal-cancer-classification-challenge-2026 \
        --ckpts /kaggle/working/runs/fold0_best.pt /kaggle/working/runs/fold1_best.pt ... \
        --out /kaggle/working/submission.csv \
        --batch-size 256 --tta
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from dataset import CellDataset, load_test_df
from model import MultimodalClassifier
from transforms import eval_modality_transform


def _d4_augments(bf: torch.Tensor, fl: torch.Tensor):
    """Yield all 8 elements of the D4 dihedral group (4 rotations × 2 reflections).

    Cells have no canonical orientation, so D4 is the natural symmetry group.
    The original (k=0, no flip) is yielded first so callers can use it as the
    base and accumulate the remaining 7.
    """
    for k in range(4):
        bfr = torch.rot90(bf, k, dims=(-2, -1))
        flr = torch.rot90(fl, k, dims=(-2, -1))
        yield bfr, flr
        yield TF.hflip(bfr), TF.hflip(flr)


def predict_one_ckpt(ckpt_path: Path, loader, device: str, tta: bool) -> np.ndarray:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Infer backbone from saved args if available; default resnet18 for old ckpts.
    saved_args = state.get("args", {})
    backbone = saved_args.get("backbone", "resnet18")
    dropout  = float(saved_args.get("dropout", 0.3))
    model = MultimodalClassifier(pretrained=False, backbone=backbone, dropout=dropout).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            bf = batch["bf"].to(device, non_blocking=True)
            fl = batch["fl"].to(device, non_blocking=True)
            p = None
            aug_iter = _d4_augments(bf, fl) if tta else [(bf, fl)]
            n_aug = 8 if tta else 1
            for bf_t, fl_t in aug_iter:
                with torch.amp.autocast("cuda", enabled=device == "cuda"):
                    pi = torch.sigmoid(model(bf_t, fl_t)).float()
                p = pi if p is None else p + pi
            preds.append((p / n_aug).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_df = load_test_df(args.data_root / "sampleSubmission.csv")
    print(f"Predicting on {len(test_df)} test images.")

    ds = CellDataset(
        test_df,
        args.data_root / "BF" / "test",
        args.data_root / "FL" / "test",
        bf_transform=eval_modality_transform("bf"),
        fl_transform=eval_modality_transform("fl"),
        paired_transform=None,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    all_preds = []
    for ckpt in args.ckpts:
        print(f"  - {ckpt}")
        all_preds.append(predict_one_ckpt(ckpt, loader, device, args.tta))
    preds = np.mean(all_preds, axis=0)

    sub = pd.DataFrame({"Name": test_df["Name"].values, "Diagnosis": preds})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  (mean pred = {preds.mean():.3f})")


if __name__ == "__main__":
    main()
