"""Where does the v47_s2 (0.8355) advantage over v47 ensemble (0.8264) come from?

We don't have ground-truth test labels, but we DO have per-cell predictions
from both the v47 ensemble and v47_seed2, plus the v46 teacher predictions
that both were trained against. That lets us answer:

  1. Is the +0.0091 LB lift concentrated on a small set of cells, or spread out?
  2. Are the seed2-vs-ensemble disagreements clustered in regions where the
     v46 teacher was UNCERTAIN (i.e., the disagreement is in the "dark
     knowledge" middle band of teacher probabilities) — or in regions where
     the teacher was CONFIDENT (i.e., seed2 is overriding a confident teacher)?
  3. Per-decile breakdown by ensemble confidence: where do the two diverge most?

The methodological question this informs is §VII-G "open": is v47_seed2's
public-LB lift real signal (likely to replicate on private LB) or split-luck
(likely to regress)?

  - If divergence is concentrated in middle-band teacher cells => seed2's
    advantage is using student-side noise to break ties the teacher couldn't,
    which IS expected to generalize (more dark-knowledge utilization).
  - If divergence is concentrated in confident-teacher cells => seed2 is
    overriding the teacher in a way that's more likely to be public-split
    overfit.

Outputs:
  - Console table
  - overleaf-report/notes/v47_seed2_vs_ensemble_stats.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Force UTF-8 stdout so unicode in print() doesn't crash on Windows cp1252 consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
RESULTS = PROJECT_ROOT / "results"
OUT_CSV = HERE.parent / "notes" / "v47_seed2_vs_ensemble_stats.csv"

def load(name):
    df = pd.read_csv(RESULTS / name).sort_values("Name").reset_index(drop=True)
    return df

# === Load the three things we need ===
ens = load("v47/submission.csv")            # v47 3-seed ensemble (LB 0.8264)
s2  = load("v47/submission_seed2.csv")      # v47 seed 2 standalone (LB 0.8355)
v46 = load("v46/submission.csv")            # v46 teacher (LB 0.8236, fed to v47)

assert (ens["Name"] == s2["Name"]).all() and (ens["Name"] == v46["Name"]).all()
p_ens = ens["Diagnosis"].values
p_s2  = s2["Diagnosis"].values
p_v46 = v46["Diagnosis"].values

n = len(p_ens)
delta = p_s2 - p_ens
agree_sign = (p_ens > 0.5) == (p_s2 > 0.5)
flip_rate  = (~agree_sign).mean()

print("=" * 78)
print("v47_seed2 (LB 0.8355) vs v47 ensemble (LB 0.8264) — per-cell agreement")
print("=" * 78)
print(f"\nTotal cells: {n:,}")
print(f"Delta = p_s2 - p_ens stats:")
print(f"  mean       = {delta.mean():+.4f}")
print(f"  median     = {np.median(delta):+.4f}")
print(f"  std        = {delta.std():.4f}")
print(f"  abs mean   = {np.abs(delta).mean():.4f}")
print(f"  10th / 90th pctl = {np.percentile(delta, 10):+.4f} / {np.percentile(delta, 90):+.4f}")
print(f"  max neg / pos    = {delta.min():+.4f} / {delta.max():+.4f}")
print(f"Sign-flip rate (different predicted class): {flip_rate*100:.2f}%  ({(~agree_sign).sum():,} cells)")

# === Q1: Concentration of disagreement ===
# What fraction of cells account for what fraction of |delta|?
abs_d_sorted = np.sort(np.abs(delta))[::-1]   # largest first
cum_d = abs_d_sorted.cumsum() / abs_d_sorted.sum()
def frac_for_thr(thr):
    """Fraction of cells needed to account for thr fraction of total |delta|."""
    idx = np.searchsorted(cum_d, thr)
    return idx / n

print(f"\n--- Q1: How concentrated is the disagreement? ---")
print(f"  Top 1%  of cells account for {cum_d[int(0.01*n)]*100:5.1f}% of total |delta|")
print(f"  Top 5%  of cells account for {cum_d[int(0.05*n)]*100:5.1f}% of total |delta|")
print(f"  Top 10% of cells account for {cum_d[int(0.10*n)]*100:5.1f}% of total |delta|")
print(f"  Top 25% of cells account for {cum_d[int(0.25*n)]*100:5.1f}% of total |delta|")

# === Q2: Are disagreements clustered in teacher-uncertain or teacher-confident cells? ===
# Bin cells by v46 teacher confidence and measure flip rate + mean |delta| in each bin.
v46_bins = [0.0, 0.05, 0.20, 0.40, 0.60, 0.80, 0.95, 1.01]
v46_bin_labels = ["[0.00,0.05)", "[0.05,0.20)", "[0.20,0.40)",
                  "[0.40,0.60)", "[0.60,0.80)", "[0.80,0.95)", "[0.95,1.00]"]
v46_bin_idx = np.digitize(p_v46, v46_bins) - 1
v46_bin_idx = np.clip(v46_bin_idx, 0, len(v46_bin_labels) - 1)

print(f"\n--- Q2: Disagreement by v46 teacher confidence band ---")
print(f"  (v46 teacher fed the same target to v47 ensemble and v47_s2 training)\n")
print(f"  {'Teacher prob bin':<14} {'cells':>8} {'flip %':>8} {'mean |d|':>10} {'share of |d|':>13}")
print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*10} {'-'*13}")
bin_rows = []
total_abs_delta = np.abs(delta).sum()
for i, lbl in enumerate(v46_bin_labels):
    mask = v46_bin_idx == i
    n_b = int(mask.sum())
    if n_b == 0:
        continue
    flip_b = (~agree_sign[mask]).mean() * 100
    abs_d_b = np.abs(delta[mask]).mean()
    share_b = np.abs(delta[mask]).sum() / total_abs_delta * 100
    print(f"  {lbl:<14} {n_b:>8,} {flip_b:>7.2f}% {abs_d_b:>10.4f} {share_b:>12.1f}%")
    bin_rows.append({"v46_teacher_bin": lbl, "cells": n_b,
                     "flip_pct": flip_b, "mean_abs_delta": abs_d_b,
                     "share_of_abs_delta_pct": share_b})

# === Q3: Per-decile by ensemble confidence ===
print(f"\n--- Q3: Disagreement by v47 ENSEMBLE confidence decile ---")
print(f"  (does s2 differ from ens more on cells the ens was uncertain about?)\n")
ens_deciles = pd.qcut(p_ens, q=10, labels=False, duplicates="drop")
print(f"  {'decile':<8} {'p_ens range':<20} {'cells':>8} {'flip %':>8} {'mean |d|':>10}")
print(f"  {'-'*8} {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
for d in sorted(np.unique(ens_deciles)):
    mask = ens_deciles == d
    n_b = int(mask.sum())
    lo, hi = p_ens[mask].min(), p_ens[mask].max()
    flip_b = (~agree_sign[mask]).mean() * 100
    abs_d_b = np.abs(delta[mask]).mean()
    print(f"  D{d+1:<7} [{lo:.3f}, {hi:.3f}]   {n_b:>8,} {flip_b:>7.2f}% {abs_d_b:>10.4f}")

# === Headline summary ===
mid_band_mask = (p_v46 >= 0.05) & (p_v46 <= 0.95)
share_mid_band = np.abs(delta[mid_band_mask]).sum() / total_abs_delta * 100
tail_mask = ~mid_band_mask
share_tails = np.abs(delta[tail_mask]).sum() / total_abs_delta * 100

print(f"\n=== HEADLINE FINDING ===")
print(f"v46 teacher MIDDLE BAND (0.05 ≤ p_teacher ≤ 0.95, the 'dark knowledge' cells):")
print(f"  cells:                   {mid_band_mask.sum():,} ({mid_band_mask.mean()*100:.1f}% of test set)")
print(f"  share of total |d|:      {share_mid_band:.1f}%")
print(f"v46 teacher TAILS (p < 0.05 or p > 0.95, the 'confident' cells):")
print(f"  cells:                   {tail_mask.sum():,} ({tail_mask.mean()*100:.1f}% of test set)")
print(f"  share of total |d|:      {share_tails:.1f}%")

if share_mid_band > 60:
    print(f"\n=> s2-vs-ensemble disagreement is CONCENTRATED in the teacher's dark-knowledge")
    print(f"  middle band ({share_mid_band:.0f}% of |d| from {mid_band_mask.mean()*100:.0f}% of cells).")
    print(f"  This is consistent with seed-2's lift being SIGNAL (s2 broke ties the")
    print(f"  ensemble averaged into hedged predictions), more likely to replicate on private LB.")
elif share_tails > 60:
    print(f"\n=> s2-vs-ensemble disagreement is CONCENTRATED in the teacher's confident tails")
    print(f"  ({share_tails:.0f}% of |d|). s2 is overriding a confident teacher, more likely to")
    print(f"  be public-split overfit (less expected to replicate on private LB).")
else:
    print(f"\n=> s2-vs-ensemble disagreement is DISTRIBUTED across confidence bands")
    print(f"  (no concentration in either teacher-uncertain or teacher-confident regions).")
    print(f"  Ambiguous signal — could be either generalization gain or split-specific.")

# === Save CSV ===
pd.DataFrame(bin_rows).to_csv(OUT_CSV, index=False, float_format="%.4f")
print(f"\nSaved {OUT_CSV}")
