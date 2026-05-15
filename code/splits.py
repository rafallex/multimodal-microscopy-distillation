"""
Patient-grouped cross-validation splits.

Every cell of a cancer patient is labeled cancer (weak labels). If we split
cells randomly, the model memorizes patient-specific artifacts (staining hue,
scanner pattern) and val AUC becomes wildly optimistic. We MUST keep every
cell of a given patient on one side of the split.

Reality of this dataset (12 train patients: 5 cancer, 7 healthy):
- 5-fold GroupKFold → some folds get 0 cancer patients → val AUC undefined.
- 3-fold StratifiedGroupKFold is the practical default: each fold gets
  roughly 1-2 cancer + 2-3 healthy patients, both classes always present.
- Leave-one-patient-out (12 folds) is also feasible and gives the most
  honest OOF estimate, but each fold alone has only one true label.

For the official OOF AUC, concatenate held-out predictions across all folds
and call roc_auc_score once on the union.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold,
)


def stratified_patient_kfold(
    df: pd.DataFrame, n_splits: int = 3, seed: int = 1, strict: bool = True
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) splits that:
       - never share a patient between train and val
       - keep the cancer/healthy ratio balanced across folds (as much as possible).

    Default seed=1 was verified to give every fold both classes for this
    dataset (12 patients: 5 cancer, 7 healthy). For n_splits=3, 40 of 50
    seeds work; seed=0 is bad (fold 2 has 0 cancer patients).

    If `strict`, raises if any fold has only one class - AUC would be undefined.
    """
    skgf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df["Diagnosis"].to_numpy()
    groups = df["patient_id"].to_numpy()
    splits = list(skgf.split(df, y=y, groups=groups))
    if strict:
        for f, (_, va) in enumerate(splits):
            yva = y[va]
            if len(np.unique(yva)) < 2:
                raise ValueError(
                    f"Fold {f} contains only class {int(yva[0])} - val AUC undefined. "
                    f"Try a different seed (1, 2, 3, 6, 7, 8 are known-good for n_splits=3)."
                )
    for tr, va in splits:
        yield tr, va


def patient_group_kfold(
    df: pd.DataFrame, n_splits: int = 3, seed: int = 0
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Simple grouped K-fold (no stratification). Kept for ablations."""
    rng = np.random.RandomState(seed)
    patients = df["patient_id"].to_numpy()
    unique_pats = np.unique(patients)
    rng.shuffle(unique_pats)
    order = {p: i for i, p in enumerate(unique_pats)}
    groups = np.array([order[p] for p in patients])
    gkf = GroupKFold(n_splits=n_splits)
    for tr, va in gkf.split(df, groups=groups):
        yield tr, va


def leave_one_patient_out(df: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """LOPO split: one patient = one val fold. 12 folds for this dataset.

    Per-fold AUC is undefined (val has only one class). Use this together
    with the OOF aggregation trick: concatenate all per-fold val predictions
    and call roc_auc_score once on the union.
    """
    logo = LeaveOneGroupOut()
    for tr, va in logo.split(df, groups=df["patient_id"].to_numpy()):
        yield tr, va


def summarize_split(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> str:
    """Pretty-print which patients ended up in train/val and the class balance."""
    tr, va = df.iloc[train_idx], df.iloc[val_idx]
    return (
        f"train: {len(tr):>6} cells, {tr['patient_id'].nunique():>2} patients, "
        f"pos-rate {tr['Diagnosis'].mean():.3f} "
        f"| val: {len(va):>6} cells, {va['patient_id'].nunique():>2} patients, "
        f"pos-rate {va['Diagnosis'].mean():.3f} "
        f"| val patients: {sorted(va['patient_id'].unique().tolist())}"
    )
