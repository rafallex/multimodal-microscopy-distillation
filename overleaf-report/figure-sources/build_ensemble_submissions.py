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

  G_v47_per_cell_confidence_weighted.csv
                                    — per-cell weighting by each seed's own
                                      decisiveness |p_i - 0.5|. In confident-tail
                                      cells (where all seeds agree) ~= equal
                                      weighting. In middle-band cells the most
                                      decisive seed per cell dominates. Directly
                                      motivated by the 97%-middle-band finding:
                                      lets the seed that "broke the tie" win
                                      in exactly the region where v47_s2
                                      outperformed the ensemble.

  H_B_plus_v46_ens.csv              — STACK of B (drop-worst within-recipe fix)
                                      and D (cross-recipe diversification fix).
                                      = (B + v46_ens) / 2
                                      = 0.25*v47_s2 + 0.25*v47_s3 + 0.5*v46_ens
                                      Tests whether the two §VII-G ensemble-rescue
                                      lessons from B (+0.0017) and D (+0.0053)
                                      add up. If additive, expect ~0.834;
                                      if non-additive, lands between B and D.

  I_v47_s2_70_s3_30.csv             — asymmetric within-recipe weighting:
                                      0.7*v47_s2 + 0.3*v47_s3. Drops s1
                                      entirely AND boosts s2 over s3.
                                      Compares to B (equal 50/50 of s2/s3,
                                      LB 0.8281) and F (LB-weighted softmax,
                                      s1=10%/s2=76%/s3=14%). Three weighting
                                      strategies on the same v47 seeds gives
                                      a clean test of which mechanism
                                      (drop-worst, drop-worst-then-asymmetric,
                                      LB-information) captures s2's pattern.

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

# === G: v47 per-cell confidence-weighted ensemble ===
# For each cell, each seed's weight = its own decisiveness |p - 0.5|.
# In tail cells (all seeds confident in same direction) ~= equal weighting.
# In middle-band cells, whichever seed is most decisive dominates -
# directly exploits the 97% middle-band finding from build_v47_seed_vs_ensemble_analysis.py.
eps = 1e-9
c1, c2, c3 = np.abs(p_s1 - 0.5), np.abs(p_s2 - 0.5), np.abs(p_s3 - 0.5)
csum = c1 + c2 + c3 + eps
G = (c1 * p_s1 + c2 * p_s2 + c3 * p_s3) / csum
mean_max_seed = np.argmax(np.column_stack([c1, c2, c3]), axis=1)
s1_dom = (mean_max_seed == 0).sum() / len(p_s1) * 100
s2_dom = (mean_max_seed == 1).sum() / len(p_s1) * 100
s3_dom = (mean_max_seed == 2).sum() / len(p_s1) * 100
print(f"\n  G: per-cell most-confident-seed distribution:  s1={s1_dom:.1f}%  s2={s2_dom:.1f}%  s3={s3_dom:.1f}%")
save("G_v47_per_cell_confidence_weighted.csv", G,
     "per-cell confidence-weighted (exploits 97% middle-band finding).")

# === H: stack B (drop-worst within-recipe) + D (cross-recipe at 0.012 gap) ===
# Tonight B gave +0.0017 vs v47 ens, D gave +0.0053. If both fixes are
# additive (different mechanisms), this should give roughly +0.005 to +0.008,
# i.e. land around 0.832-0.835. If non-additive, lands between B and D.
H = (B + p_v46) / 2.0
save("H_B_plus_v46_ens.csv", H,
     "stack §VII-G fix (B) + §VII-F fix (D). 0.25 v47_s2 + 0.25 v47_s3 + 0.5 v46_ens.")

# === I: asymmetric within-recipe weighting (drop s1, boost s2 vs s3) ===
# Compares to B (50/50 equal s2/s3) and F (LB-weighted softmax that includes s1
# at 10%). Three within-recipe weighting strategies on the same seeds gives a
# clean test of which mechanism captures s2's middle-band advantage.
I = 0.7 * p_s2 + 0.3 * p_s3
save("I_v47_s2_70_s3_30.csv", I,
     "asymmetric within-recipe: 0.7*v47_s2 + 0.3*v47_s3 (drop s1, boost s2).")

# === Summary stats for the paper ===
print()
print("Per-cell agreement summary (vs v47_s2 = the score-anchor):")
print(f"  {'submission':<30} {'mean delta':>12} {'pearson r':>12} {'flip %':>8}")
print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")
for name, p, lbl in [("B_v47_top2_mean", B, "B"),
                     ("C_v47_per_cell_median", C, "C"),
                     ("D_v47s2_plus_v46_ens", D, "D"),
                     ("E_v47s2_plus_v44s1", E, "E"),
                     ("F_v47_lb_weighted", F, "F"),
                     ("G_v47_per_cell_confidence_weighted", G, "G"),
                     ("H_B_plus_v46_ens", H, "H"),
                     ("I_v47_s2_70_s3_30", I, "I")]:
    delta = (p - p_s2).mean()
    r = np.corrcoef(p, p_s2)[0, 1]
    flip = ((p > 0.5) != (p_s2 > 0.5)).mean() * 100
    print(f"  {name:<30} {delta:>+12.4f} {r:>12.4f} {flip:>7.2f}%")

print()
print("=" * 78)
print("UPLOAD ORDER (B/C/D/E submitted 2026-05-27 evening):")
print("  B = 0.8281 (+0.0017 vs v47 ens) — drop-worst marginal")
print("  C = 0.8201 (-0.0063) — median actively hurts (negative result)")
print("  D = 0.8317 (+0.0053) — cross-recipe at 0.012 gap WINS")
print("  E = 0.8161 (-0.0103) — wide-gap confirms §VII-A drag")
print()
print("TOMORROW (May 29) 4 slots:")
print("  Slot 1: G (per-cell confidence-weighted, exploits 97% middle-band)")
print("  Slot 2: H (stack B+D fixes, tests additivity)")
print("  Slot 3: F (LB-weighted, §VII-A risk test)")
print("  Slot 4: I (asymmetric within-recipe 0.7/0.3, compare to B and F)")
print("=" * 78)
print(f"\nAll 5 CSVs written to: {OUT_DIR}")
