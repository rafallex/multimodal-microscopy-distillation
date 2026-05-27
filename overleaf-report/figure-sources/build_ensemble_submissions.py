"""Build 5 CPU-only ensemble submissions to use the daily Kaggle quota productively.

We have ~12 free Kaggle scoring slots (4/day x 3 days) before the May 30 GPU
quota reset. Each output here is a recombination of existing per-seed / per-recipe
submission CSVs — no model retraining required.

Each submission tests a specific methodological hypothesis that strengthens
the paper, even if it underperforms v47_seed2 on the public LB.

Outputs (all in results/cpu_ensembles/):

  B_v47_top2_mean.csv               — per-cell mean of v47 seeds 2 + 3 only
                                      (drops the worst seed s1). Tests
                                      §VII-G "drop-worst rescues ensemble".

  C_v47_per_cell_median.csv         — per-cell median across all 3 v47 seeds.
                                      Robust-statistic variant of B.

  D_v47s2_plus_v46_ens.csv          — sigmoid-average of v47_s2 + v46 ensemble.
                                      Cross-recipe at 0.012 LB gap — adds a 3rd
                                      data point to the §VII-F threshold curve
                                      (v45_probe at 0.025 gap, v22 at 0.05).

  E_v47s2_plus_v44s1.csv            — sigmoid-average of v47_s2 + v44_seed1.
                                      Cross-recipe at 0.051 gap — expected to
                                      confirm §VII-A failure at wide gaps.

  F_v47_lb_weighted.csv             — softmax-weighted ensemble of v47 seeds,
                                      weights = softmax((LB_i - mean_LB)/0.01).
                                      Uses LB info to weight — slight overfit
                                      risk per §VII-A diagnosis; defensible
                                      as one test, suspect as a strategy.

Each file has the Kaggle-required (Name, Diagnosis) columns. Upload directly.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
RESULTS = PROJECT_ROOT / "results"
OUT_DIR = RESULTS / "cpu_ensembles"
OUT_DIR.mkdir(exist_ok=True)

# Known per-seed public LBs (used for weight F)
LB_V47 = {1: 0.8150, 2: 0.8355, 3: 0.8187}

def load(path):
    df = pd.read_csv(RESULTS / path).sort_values("Name").reset_index(drop=True)
    return df

# Load everything we need
v47_s1 = load("v47/submission_seed1.csv")
v47_s2 = load("v47/submission_seed2.csv")
v47_s3 = load("v47/submission_seed3.csv")
v46_ens = load("v46/submission.csv")
v44_s1  = load("v44/submission_seed1.csv")

# Sanity: all CSVs cover identical Name index
for df, name in [(v47_s2, "v47_s2"), (v47_s3, "v47_s3"),
                 (v46_ens, "v46_ens"), (v44_s1, "v44_s1")]:
    assert (v47_s1["Name"] == df["Name"]).all(), f"Name mismatch: v47_s1 vs {name}"

names = v47_s1["Name"]
p_s1, p_s2, p_s3 = v47_s1["Diagnosis"].values, v47_s2["Diagnosis"].values, v47_s3["Diagnosis"].values
p_v46  = v46_ens["Diagnosis"].values
p_v44s1 = v44_s1["Diagnosis"].values

def save(name, p, description):
    out = pd.DataFrame({"Name": names, "Diagnosis": p})
    path = OUT_DIR / name
    out.to_csv(path, index=False)
    print(f"  {name:<38} mean={p.mean():.4f}  sd={p.std():.4f}  | {description}")
    return path

print("=" * 78)
print("Building CPU-only ensemble submissions in results/cpu_ensembles/")
print("=" * 78)
print()

# === B: v47 top-2 mean (drop worst seed s1) ===
B = (p_s2 + p_s3) / 2.0
save("B_v47_top2_mean.csv", B,
     "drop-worst-seed test (§VII-G). Per-cell mean of s2 + s3 only.")

# === C: v47 per-cell median across 3 seeds ===
C = np.median(np.column_stack([p_s1, p_s2, p_s3]), axis=1)
save("C_v47_per_cell_median.csv", C,
     "robust-stat variant (§VII-G). Per-cell median of 3 seeds.")

# === D: v47_s2 + v46 ensemble sigmoid-avg (cross-recipe, gap=0.012) ===
D = (p_s2 + p_v46) / 2.0
save("D_v47s2_plus_v46_ens.csv", D,
     "cross-recipe at 0.012 LB gap (§VII-F). v47_s2 + v46 ensemble.")

# === E: v47_s2 + v44_s1 sigmoid-avg (cross-recipe, gap=0.051) ===
E = (p_s2 + p_v44s1) / 2.0
save("E_v47s2_plus_v44s1.csv", E,
     "cross-recipe at 0.051 LB gap (§VII-A). v47_s2 + v44_seed1.")

# === F: v47 LB-weighted ensemble (softmax with T=0.01) ===
seed_lbs = np.array([LB_V47[1], LB_V47[2], LB_V47[3]])
T = 0.01
logits = (seed_lbs - seed_lbs.mean()) / T
weights = np.exp(logits) / np.exp(logits).sum()
print(f"\n  F weights: s1={weights[0]:.3f}  s2={weights[1]:.3f}  s3={weights[2]:.3f}  (softmax T=0.01)")
F = weights[0] * p_s1 + weights[1] * p_s2 + weights[2] * p_s3
save("F_v47_lb_weighted.csv", F,
     f"LB-weighted (§VII-A risk). weights s1={weights[0]:.2f}/s2={weights[1]:.2f}/s3={weights[2]:.2f}.")

# === Summary stats for the paper ===
print()
print("Per-cell agreement summary (vs v47_s2 = the score-anchor):")
print(f"  {'submission':<30} {'mean delta':>12} {'pearson r':>12} {'flip %':>8}")
print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")
for name, p, lbl in [("B_v47_top2_mean", B, "B"),
                     ("C_v47_per_cell_median", C, "C"),
                     ("D_v47s2_plus_v46_ens", D, "D"),
                     ("E_v47s2_plus_v44s1", E, "E"),
                     ("F_v47_lb_weighted", F, "F")]:
    delta = (p - p_s2).mean()
    r = np.corrcoef(p, p_s2)[0, 1]
    flip = ((p > 0.5) != (p_s2 > 0.5)).mean() * 100
    print(f"  {name:<30} {delta:>+12.4f} {r:>12.4f} {flip:>7.2f}%")

print()
print("=" * 78)
print("UPLOAD ORDER (max paper value first):")
print("  Tomorrow (May 28): B, C, D, F  — 4 slots, 4 hypotheses")
print("  May 29:            E + retries / variants if any landed unexpectedly")
print("=" * 78)
print(f"\nAll 5 CSVs written to: {OUT_DIR}")
