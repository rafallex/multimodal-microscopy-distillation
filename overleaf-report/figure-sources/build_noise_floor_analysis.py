"""Noise-floor and per-cell stability analysis for the headline LB deltas.

We do NOT have ground-truth labels for the test set (private Kaggle), so we
cannot bootstrap AUC directly. Instead we characterize the noise floor and
prediction stability with the data we DO have:

  1. Per-seed public-LB stats for v47 (the only version with 3 standalone-submitted
     seeds). Gives an empirical SD / SE / percentile CI of the recipe-level noise.
  2. Per-cell prediction-difference SD across v47's 3 seeds — an intrinsic
     "how much does seed alone move per-cell predictions" measurement
     that is independent of any LB / ground-truth signal.
  3. Pearson r between ensemble predictions for each headline pair
     (v19/v41, v41/v44, v44/v46, v46/v47, plus all-pairs reference).
  4. Per-cell sign-flip rate (cells where one ensemble says >0.5 and the other <0.5)
     for each headline pair — characterizes how much the predicted class
     distribution actually changes between versions.
  5. Each headline LB delta expressed as a multiple of the v47-derived per-seed SE.
     This is the closest defensible analog to a t-statistic without ground truth.

Outputs:
  - Console table with all numbers
  - overleaf-report/notes/noise_floor_stats.csv (machine-readable)
  - overleaf-report/notes/noise_floor_table.tex (latex fragment for §VIII-E)
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
RESULTS = PROJECT_ROOT / "results"
OUT_CSV = HERE.parent / "notes" / "noise_floor_stats.csv"      # local-only, machine-readable
OUT_TEX = HERE.parent / "noise_floor_table.tex"                 # part of Overleaf upload (\input from main.tex)

# === Known per-seed public LB values (from Kaggle submissions) ===
LB = {
    "v19":        0.7455,
    "v41":        0.7563,
    "v44":        0.7812,   # ensemble
    "v44_seed1":  0.7844,   # only seed submitted standalone
    "v46":        0.8236,   # ensemble
    "v46_seed1":  0.8157,
    "v46_seed2":  0.8229,
    "v46_seed3":  0.8187,   # discovered 2026-05-27 in full Kaggle submission list
    "v47":        0.8264,   # ensemble
    "v47_seed1":  0.8150,
    "v47_seed2":  0.8355,
    "v47_seed3":  0.8126,   # the 0.8187 reported on 2026-05-27 was actually v46_s3 (file collision: both named submission_seed3.csv)
}

HEADLINE_PAIRS = [
    ("v19", "v41", +0.0108),   # +0.011 in paper, computed: 0.7563-0.7455
    ("v41", "v44", +0.0249),   # +0.025
    ("v44", "v46", +0.0424),   # +0.042 (vs v44 ens) / +0.0392 (vs v44_seed1)
    ("v44_seed1", "v46", +0.0392),
    ("v46", "v47", +0.0028),   # +0.003 ensemble
    ("v46", "v47_seed2", +0.0119),  # +0.012 best seed
]

def load_csv(name):
    """Load a submission CSV by short name. Returns predictions sorted by Name."""
    if "_seed" in name:
        v, s = name.split("_seed")
        path = RESULTS / v / f"submission_seed{s}.csv"
    else:
        path = RESULTS / name / "submission.csv"
    df = pd.read_csv(path).sort_values("Name").reset_index(drop=True)
    return df["Diagnosis"].values

# === 1. Per-seed LB stats for v47 (n=3) and v46 (now also n=3) ===
v47_seeds = np.array([LB["v47_seed1"], LB["v47_seed2"], LB["v47_seed3"]])
v47_mean = v47_seeds.mean()
v47_sd   = v47_seeds.std(ddof=1)
v47_se   = v47_sd / np.sqrt(3)
v47_range = v47_seeds.max() - v47_seeds.min()

v46_seeds = np.array([LB["v46_seed1"], LB["v46_seed2"], LB["v46_seed3"]])
v46_mean = v46_seeds.mean()
v46_sd   = v46_seeds.std(ddof=1)
v46_se   = v46_sd / np.sqrt(3)
v46_range = v46_seeds.max() - v46_seeds.min()

# Bootstrap percentile CI for v47 (n=3 is small but instructive)
rng = np.random.default_rng(seed=42)
n_boot = 10_000
boots = np.array([rng.choice(v47_seeds, size=3, replace=True).mean() for _ in range(n_boot)])
v47_boot_lo, v47_boot_hi = np.percentile(boots, [2.5, 97.5])

# === 2. Per-cell prediction-difference SD across v47 seeds (no ground truth needed) ===
v47_s1_p = load_csv("v47_seed1")
v47_s2_p = load_csv("v47_seed2")
v47_s3_p = load_csv("v47_seed3")
per_cell_sd = np.std(np.column_stack([v47_s1_p, v47_s2_p, v47_s3_p]), axis=1, ddof=1)
mean_per_cell_sd = per_cell_sd.mean()
median_per_cell_sd = np.median(per_cell_sd)

# Fraction of cells where any pair of seeds disagrees on sigmoid sign (>0.5 vs <0.5)
flip_s1_s2 = ((v47_s1_p > 0.5) != (v47_s2_p > 0.5)).mean()
flip_s1_s3 = ((v47_s1_p > 0.5) != (v47_s3_p > 0.5)).mean()
flip_s2_s3 = ((v47_s2_p > 0.5) != (v47_s3_p > 0.5)).mean()
mean_within_v47_flip = (flip_s1_s2 + flip_s1_s3 + flip_s2_s3) / 3

# === 3 & 4. Pearson r and per-cell sign-flip rate for headline pairs ===
versions_to_load = ["v19", "v41", "v44", "v44_seed1", "v46", "v47", "v47_seed2"]
preds = {v: load_csv(v) for v in versions_to_load}

pair_stats = []
for a, b, delta in HEADLINE_PAIRS:
    r = np.corrcoef(preds[a], preds[b])[0, 1]
    flip = ((preds[a] > 0.5) != (preds[b] > 0.5)).mean()
    # SE of delta: conservative, treat each version's LB as having the v47 per-seed SE
    # (this is loose for ensembles since ensemble SE < single-seed SE, but errs toward not over-claiming)
    delta_se = v47_se * np.sqrt(2)  # SD of a difference of two indep observations each with SE=v47_se
    t_eq = delta / delta_se
    pair_stats.append({
        "pair": f"{a} -> {b}",
        "delta_lb": delta,
        "pearson_r": r,
        "sign_flip_rate": flip,
        "delta_over_se": t_eq,
        "verdict": (
            "robust"          if abs(t_eq) >= 4.0 else
            "supported"       if abs(t_eq) >= 2.0 else
            "marginal"        if abs(t_eq) >= 1.0 else
            "indistinguishable"
        ),
    })

# === Print human-readable summary ===
print("=" * 78)
print("Noise-floor analysis (no ground-truth labels available)")
print("=" * 78)

print("\n--- v47 per-seed public LB (n=3) ---")
print(f"  seeds: {v47_seeds.tolist()}")
print(f"  mean  = {v47_mean:.4f}")
print(f"  SD    = {v47_sd:.4f}  (sample, ddof=1)")
print(f"  SE    = {v47_se:.4f}  (SD/sqrt(3))")
print(f"  range = {v47_range:.4f}")
print(f"  bootstrap 95% CI of mean (n_boot={n_boot:,}): "
      f"[{v47_boot_lo:.4f}, {v47_boot_hi:.4f}]")

print(f"\n--- v46 per-seed public LB (n=3) ---")
print(f"  seeds: {v46_seeds.tolist()}")
print(f"  mean  = {v46_mean:.4f}")
print(f"  SD    = {v46_sd:.4f}  (sample, ddof=1)")
print(f"  SE    = {v46_se:.4f}  (SD/sqrt(3))")
print(f"  range = {v46_range:.4f}")
print(f"  v47 SE / v46 SE = {v47_se/v46_se:.2f}x  (v47 seed noise is {v47_se/v46_se:.1f}x larger)")
print(f"  v47 range / v46 range = {v47_range/v46_range:.2f}x wider")

print("\n--- v47 within-recipe per-cell prediction stability ---")
print(f"  mean per-cell SD across 3 seeds      = {mean_per_cell_sd:.4f}")
print(f"  median per-cell SD                   = {median_per_cell_sd:.4f}")
print(f"  mean cross-seed sign-flip rate       = {mean_within_v47_flip*100:.2f}%")
print(f"    (pairs: s1-s2 {flip_s1_s2*100:.2f}%, "
      f"s1-s3 {flip_s1_s3*100:.2f}%, s2-s3 {flip_s2_s3*100:.2f}%)")

print("\n--- Headline-pair stats ---")
print(f"  noise-floor SE_delta (conservative) = sqrt(2)*SE_v47 = {v47_se*np.sqrt(2):.4f}")
print(f"  verdict thresholds: |delta/SE|>=4 robust, >=2 supported, >=1 marginal, <1 indistinguishable\n")
hdr = f"  {'pair':<22} {'delta':>8} {'r':>7} {'flip%':>7} {'|d/SE|':>8} {'verdict':>18}"
print(hdr); print("  " + "-"*len(hdr))
for r in pair_stats:
    print(f"  {r['pair']:<22} {r['delta_lb']:>+8.4f} {r['pearson_r']:>7.3f} "
          f"{r['sign_flip_rate']*100:>6.2f}% {r['delta_over_se']:>+8.2f} {r['verdict']:>18}")

# === Write machine-readable CSV ===
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(pair_stats).to_csv(OUT_CSV, index=False, float_format="%.4f")
print(f"\nSaved {OUT_CSV}")

# === Write LaTeX table fragment for §VIII-E ===
tex = []
tex.append("% Auto-generated by overleaf-report/figure-sources/build_noise_floor_analysis.py")
tex.append("% Headline-delta significance test against the v47 per-seed noise floor.")
tex.append(r"\begin{table}[!t]")
tex.append(r"  \centering")
tex.append(r"  \caption{Headline LB deltas with prediction-level stability statistics. "
           rf"$\mathrm{{SE}}_\Delta = \sqrt{{2}} \cdot \mathrm{{SE}}_{{v47}}$ "
           rf"$= {v47_se*np.sqrt(2):.4f}$, "
           f"using the v47 per-seed SE ({v47_se:.4f}) as a conservative "
           "single-seed noise estimate. Pearson $r$ is between ensemble probability "
           "outputs over the 59{,}040 test cells; flip\\% is the fraction of cells "
           "where the sigmoid sign disagrees between the two versions.}")
tex.append(r"  \label{tab:noisefloor}")
tex.append(r"  \footnotesize")
tex.append(r"  \begin{tabular}{@{}lrrrrl@{}}")
tex.append(r"    \toprule")
tex.append(r"    Pair & $\Delta$LB & Pearson $r$ & Flip\% & $|\Delta|/\mathrm{SE}_\Delta$ & Verdict \\")
tex.append(r"    \midrule")
verdict_map = {
    "robust":            r"\textbf{robust} ($\ge 4\sigma$)",
    "supported":         r"supported ($\ge 2\sigma$)",
    "marginal":          r"marginal ($\ge 1\sigma$)",
    "indistinguishable": r"indistinguishable ($<1\sigma$)",
}
for r in pair_stats:
    pair_esc = r["pair"].replace("_", r"\_").replace(" -> ", r" $\to$ ")
    tex.append(
        f"    {pair_esc} & ${r['delta_lb']:+.4f}$ & ${r['pearson_r']:.3f}$ & "
        f"${r['sign_flip_rate']*100:.2f}\\%$ & ${r['delta_over_se']:+.2f}$ & "
        f"{verdict_map[r['verdict']]} \\\\"
    )
tex.append(r"    \bottomrule")
tex.append(r"  \end{tabular}")
tex.append(r"\end{table}")

OUT_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")
print(f"Saved {OUT_TEX}")
