"""
vEnsemble_csv.py — standalone local ensemble of Kaggle submission.csv files

Usage:
    python vEnsemble_csv.py path/to/submission_v19.csv path/to/submission_v21.csv \
        [path/to/submission_v22.csv ...] \
        [--weights 0.7455 0.76] [--out-dir ./ensemble_outputs]

What it does:
    Loads each submission.csv (must have columns: Name, Diagnosis).
    Reindexes all to match the first file's order.
    Writes three combination strategies into out-dir:
        submission_sigmoid_avg.csv   (weighted arithmetic mean of sigmoid outputs)
        submission_rank_avg.csv      (weighted average of fractional ranks)
        submission_geomean.csv       (weighted geometric mean of sigmoid outputs)

Why three strategies:
    AUC is rank-based but practical AUC on real test sets can be sensitive to how the
    component models are calibrated. Try all three and submit the best.

Tips:
    * For 2-model ensembles, sigmoid_avg usually wins (well-tested).
    * For 3+ models with different LBs, weight by LB: --weights 0.74 0.76 0.78
    * If one model is decisive (bimodal) and another is uncertain (unimodal), geomean
      tends to preserve the decisive one's extremes — try it.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def ensemble_submissions(csv_paths: list[str],
                          weights: list[float] | None = None,
                          out_dir: str = "./ensemble_outputs") -> dict[str, np.ndarray]:
    """Combine N submission CSVs into three ensemble outputs.

    Returns the dict of per-strategy prediction arrays for further analysis if needed.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not csv_paths:
        raise ValueError("Need at least one CSV path")
    print(f"\nLoading {len(csv_paths)} submission CSVs:")
    dfs: list[pd.DataFrame] = []
    for p in csv_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        df = pd.read_csv(p)
        if "Name" not in df.columns or "Diagnosis" not in df.columns:
            raise ValueError(f"{p}: expected columns 'Name' and 'Diagnosis', got {list(df.columns)}")
        print(f"  {p}  rows={len(df)}  mean={df['Diagnosis'].mean():.4f}  "
              f"std={df['Diagnosis'].std():.4f}")
        dfs.append(df)

    # Use the first CSV's Name ordering; reindex others to match it.
    names = dfs[0]["Name"].tolist()
    for i in range(1, len(dfs)):
        if dfs[i]["Name"].tolist() != names:
            print(f"  [reindex] {csv_paths[i]} to match {csv_paths[0]} order")
            dfs[i] = dfs[i].set_index("Name").reindex(names).reset_index()

    P = np.stack([d["Diagnosis"].values.astype(float) for d in dfs], axis=0)
    if np.any(np.isnan(P)):
        raise ValueError("NaN found in predictions after reindex; CSV Name lists don't match")

    M = P.shape[0]
    if weights is None:
        w = np.ones(M) / M
    else:
        if len(weights) != M:
            raise ValueError(f"Got {len(weights)} weights for {M} CSVs")
        w = np.array(weights, dtype=float)
        w = w / w.sum()
    print(f"\nNormalized weights: {w.tolist()}")

    # 1. Sigmoid arithmetic mean
    sigmoid_avg = (P * w[:, None]).sum(axis=0)

    # 2. Rank average
    R = np.zeros_like(P)
    for i in range(M):
        R[i] = rankdata(P[i], method="average") / len(P[i])
    rank_avg = (R * w[:, None]).sum(axis=0)

    # 3. Geometric mean of sigmoid outputs (with log-space for numerical stability)
    eps = 1e-7
    geo = np.exp((np.log(np.clip(P, eps, 1 - eps)) * w[:, None]).sum(axis=0))

    results: dict[str, np.ndarray] = {
        "sigmoid_avg": sigmoid_avg,
        "rank_avg":    rank_avg,
        "geomean":     geo,
    }
    print(f"\nEnsemble combination stats:")
    print(f"  {'strategy':>12}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}  "
          f"{'<0.05':>7}  {'>0.95':>7}")
    for name, p in [(f"single_{i}", P[i]) for i in range(M)] + list(results.items()):
        print(f"  {name:>12}  {p.mean():.4f}  {p.std():.4f}  {p.min():.4f}  "
              f"{p.max():.4f}  {(p<0.05).mean():>6.2%}  {(p>0.95).mean():>6.2%}")

    print(f"\nWriting outputs to {out_dir}:")
    for name, preds in results.items():
        sub = pd.DataFrame({"Name": names, "Diagnosis": preds})
        path = os.path.join(out_dir, f"submission_{name}.csv")
        sub.to_csv(path, index=False)
        print(f"  {path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble multiple Kaggle submission.csv files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_paths", nargs="+",
                        help="Paths to submission.csv files (2+ recommended)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Per-CSV weights (will be normalized). Default = equal.")
    parser.add_argument("--out-dir", default="./ensemble_outputs",
                        help="Where to write the three submission_*.csv outputs")
    args = parser.parse_args()
    ensemble_submissions(args.csv_paths, args.weights, args.out_dir)


if __name__ == "__main__":
    main()
