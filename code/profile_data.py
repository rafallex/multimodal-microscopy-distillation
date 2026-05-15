"""
Profile the dataset before training. Run this once and read the output.

Checks:
  - Number of patients in train, class balance overall, per-patient cell counts.
  - File pairing: every BF/train file has a matching FL/train file, same for test.
  - Image stats: min/max/mean/std per modality on a small random sample.

Usage:
    python profile_data.py --data-root <path-to-extracted-dataset>
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dataset import load_train_df, load_test_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--sample-size", type=int, default=500)
    args = ap.parse_args()

    root = args.data_root
    train_df = load_train_df(root / "train.csv")
    test_df = load_test_df(root / "sampleSubmission.csv")

    print("=" * 70)
    print("TRAIN")
    print("=" * 70)
    print(f"rows                : {len(train_df)}")
    print(f"unique patients     : {train_df['patient_id'].nunique()}")
    print(f"overall pos rate    : {train_df['Diagnosis'].mean():.4f}")
    print("\nPer-patient summary:")
    per_pat = train_df.groupby("patient_id").agg(
        n_cells=("Name", "size"),
        label=("Diagnosis", "first"),
        pos_rate=("Diagnosis", "mean"),
    ).reset_index()
    # Cross-check: in this dataset every cell of a patient must share the label.
    assert (per_pat["pos_rate"].isin([0.0, 1.0])).all(), \
        "Some patients have mixed labels - that shouldn't happen!"
    print(per_pat.to_string(index=False))
    print(f"\ncancer patients : {(per_pat['label'] == 1).sum()}")
    print(f"healthy patients: {(per_pat['label'] == 0).sum()}")

    print("\n" + "=" * 70)
    print("TEST")
    print("=" * 70)
    print(f"rows : {len(test_df)}")

    print("\n" + "=" * 70)
    print("BF <-> FL file pairing")
    print("=" * 70)
    for split in ["train", "test"]:
        bf = set(p.name for p in (root / "BF" / split).glob("*.jpg"))
        fl = set(p.name for p in (root / "FL" / split).glob("*.jpg"))
        only_bf = bf - fl
        only_fl = fl - bf
        print(f"{split:>5}: BF={len(bf)} FL={len(fl)} | only_BF={len(only_bf)} only_FL={len(only_fl)}")
        if only_bf:
            print(f"   sample only_BF: {list(only_bf)[:3]}")
        if only_fl:
            print(f"   sample only_FL: {list(only_fl)[:3]}")

    print("\n" + "=" * 70)
    print(f"Image stats over {args.sample_size} random TRAIN samples")
    print("=" * 70)
    random.seed(0)
    sample_names = random.sample(list(train_df["Name"]), min(args.sample_size, len(train_df)))
    for modality in ["BF", "FL"]:
        arr = []
        for n in sample_names:
            img = np.asarray(Image.open(root / modality / "train" / n).convert("L"), dtype=np.float32) / 255.0
            arr.append(img)
        a = np.stack(arr)
        print(f"{modality}: shape={a.shape}  min={a.min():.3f} max={a.max():.3f} "
              f"mean={a.mean():.3f} std={a.std():.3f}")
    print("\n(use these mean/std values to update BF_MEAN, BF_STD, FL_MEAN, FL_STD in transforms.py)")


if __name__ == "__main__":
    main()
