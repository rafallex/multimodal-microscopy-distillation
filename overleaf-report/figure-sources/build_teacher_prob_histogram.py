"""Build Fig. 3 — teacher-probability histogram with hard-threshold annotations.

Shows the distribution of v46 ensemble predictions over all 59,040 test cells.
Annotates the p<0.05 and p>0.95 hard-pseudo thresholds, the tails (HARD kept),
and the middle band that soft pseudo-labeling uses ("dark knowledge").

Outputs:
  presentation/figures/teacher_prob_histogram.png  (used by the PPT)
  overleaf-report/teacher_prob_histogram.pdf       (used by the LaTeX paper)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent              # overleaf-report/figure-sources/
PROJECT_ROOT = HERE.parent.parent                    # A3/
TEACHER_CSV = PROJECT_ROOT / "results" / "v46" / "submission.csv"
OUT_PNG = PROJECT_ROOT / "presentation" / "figures" / "teacher_prob_histogram.png"
OUT_PDF = PROJECT_ROOT / "overleaf-report" / "teacher_prob_histogram.pdf"

# Palette consistent with the deck (Microscopy Indigo)
NAVY    = "#1A3A52"
TEAL    = "#0D9488"
AMBER   = "#D97706"
CRIMSON = "#B91C1C"
CREAM   = "#FAF7F2"
MUTED   = "#64748B"

# === Load teacher probabilities ===
df = pd.read_csv(TEACHER_CSV)
p = df["Diagnosis"].values
total = len(p)
n_low  = int((p < 0.05).sum())
n_high = int((p > 0.95).sum())
n_hard = n_low + n_high
n_soft = total - n_hard

print(f"Teacher (v46): {total} test cells")
print(f"  p < 0.05:  {n_low:>6,} cells  ({n_low/total*100:.1f}%)")
print(f"  p > 0.95:  {n_high:>6,} cells  ({n_high/total*100:.1f}%)")
print(f"  total HARD kept (tails): {n_hard:,}  ({n_hard/total*100:.1f}%)")
print(f"  SOFT-only middle band:   {n_soft:,}  ({n_soft/total*100:.1f}%)")
print(f"  mean: {p.mean():.4f}  median: {np.median(p):.4f}")

# === Figure ===
fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
fig.patch.set_facecolor("white")

# Histogram with 60 bins
bins = np.linspace(0, 1, 61)
counts, edges, patches = ax.hist(p, bins=bins, color=TEAL, alpha=0.75,
                                  edgecolor="white", linewidth=0.6)

# Find max bar height for placing annotations
ymax = counts.max() * 1.20
ax.set_ylim(0, ymax)
ax.set_xlim(-0.01, 1.01)

# Hard-threshold dashed lines
for thr, lbl in [(0.05, "p = 0.05"), (0.95, "p = 0.95")]:
    ax.axvline(thr, color=CRIMSON, linestyle="--", linewidth=1.5, alpha=0.85,
               zorder=3)
    # Label above the line
    ax.text(thr, ymax * 0.98, f" {lbl} ", color=CRIMSON, fontsize=9,
            fontweight="bold", ha="left" if thr < 0.5 else "right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=CRIMSON, linewidth=1.0, alpha=0.95))

# Shade the tails (HARD kept zones) with light crimson
ax.axvspan(-0.01, 0.05, color=CRIMSON, alpha=0.08, zorder=0)
ax.axvspan(0.95, 1.01, color=CRIMSON, alpha=0.08, zorder=0)

# "HARD kept" labels under the tails
ax.text(0.025, ymax * 0.05, "HARD kept", color=CRIMSON,
        fontsize=9, fontweight="bold", ha="center", va="bottom")
ax.text(0.975, ymax * 0.05, "HARD kept", color=CRIMSON,
        fontsize=9, fontweight="bold", ha="center", va="bottom")

# Big amber annotation over the middle band: "SOFT keeps all of this"
# Position over the middle ~70% of x-axis
ax.annotate("",
            xy=(0.05, ymax * 0.74),
            xytext=(0.95, ymax * 0.74),
            arrowprops=dict(arrowstyle="<->", color=AMBER, lw=2.0),
            zorder=4)
ax.text(0.50, ymax * 0.82,
        f"SOFT keeps all of this — {n_soft:,} cells of dark knowledge",
        color=AMBER, fontsize=11, fontweight="bold",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=CREAM,
                  edgecolor=AMBER, linewidth=1.2))

# Axis styling
ax.set_xlabel("Teacher probability · p(malignant)",
              fontsize=11, color=NAVY)
ax.set_ylabel("Test cells (count)", fontsize=11, color=NAVY)
ax.tick_params(axis="both", colors=MUTED, labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.spines["left"].set_color(MUTED)
ax.grid(True, axis="y", alpha=0.25, linestyle=":")

# Title (small, top-left)
ax.set_title(
    f"Distribution of v46 teacher predictions over {total:,} test cells",
    fontsize=12, fontweight="bold", color=NAVY, pad=10, loc="left")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
print(f"\nSaved {OUT}")
print(f"Saved {OUT.with_suffix('.pdf')}")
print(f"PNG size: {OUT.stat().st_size / 1024:.1f} KB")
