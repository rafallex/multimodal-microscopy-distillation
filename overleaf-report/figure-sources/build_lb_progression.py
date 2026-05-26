"""Build the LB-progression figure for the A3 report and presentation.

Plots public LB vs chronological version order, annotates the key milestones,
and adds horizontal reference lines for the v19 baseline and the prior leader.

Outputs:
  presentation/figures/lb_progression.png  (300 dpi, ~1600x900 px, used by the PPT)
  overleaf-report/lb_progression.pdf        (used by the LaTeX paper)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# === Data from LB_HISTORY.md (chronological order, only submitted runs) ===
# (version_label, lb, marker_color, annotate_offset_xy_or_None)
RUNS = [
    ("v19",          0.7455, "tab:gray",    None),
    ("v20",          0.5974, "tab:gray",    None),
    ("v21",          0.7018, "tab:gray",    None),
    ("v22\n(ensemble)", 0.7422, "tab:red",  (-0.5, -0.04)),
    ("v23",          0.7154, "tab:gray",    None),
    ("v30",          0.6448, "tab:red",     (0.3, -0.025)),
    ("v34",          0.7155, "tab:gray",    None),
    ("v37",          0.7092, "tab:red",     None),
    ("v38",          0.7147, "tab:gray",    None),
    ("v41",          0.7563, "tab:blue",    (0.2, 0.012)),
    ("v42",          0.5908, "tab:red",     (0.3, -0.02)),
    ("v43",          0.7444, "tab:red",     (0.0, -0.025)),
    ("v44",          0.7812, "tab:green",   (-0.2, 0.012)),
    ("v44_seed1",    0.7844, "tab:green",   None),
    ("v45_probe",    0.7729, "tab:red",     None),
    ("v46_seed1",    0.8157, "darkgreen",   None),
    ("v46_seed2",    0.8229, "darkgreen",   None),
    ("v46",          0.8236, "tab:green",   (-1.1, 0.015)),
    ("v47",          0.8264, "tab:green",   (0.5, 0.012)),
    ("v47_s1",       0.8150, "#0E5C2F",     None),  # darker green for v47 per-seed extracts
    ("v47_s2",       0.8355, "#0E5C2F",     None),  # ← best single seed across the whole project
    ("v47_s3",       0.8126, "#0E5C2F",     None),
]
# v48 placeholder (queued — Hinton T=2 temperature distillation)
V48_PLACEHOLDER = ("v48", None, "tab:orange")

PRIOR_LEADER = 0.7916  # was the public LB top before v46 took #1
V19_BASELINE = 0.7455

HERE = Path(__file__).resolve().parent              # overleaf-report/figure-sources/
PROJECT_ROOT = HERE.parent.parent                    # A3/
OUT_PNG = PROJECT_ROOT / "presentation" / "figures" / "lb_progression.png"
OUT_PDF = PROJECT_ROOT / "overleaf-report" / "lb_progression.pdf"

# === Figure ===
fig, ax = plt.subplots(figsize=(13, 6.5), dpi=120)

xs = list(range(len(RUNS)))
ys = [r[1] for r in RUNS]
colors = [r[2] for r in RUNS]
labels = [r[0] for r in RUNS]

# Connect with a thin grey line for visual flow
ax.plot(xs, ys, color="#cccccc", linewidth=1, zorder=1)

# Per-point markers colored by outcome category
for x, y, c, lab in zip(xs, ys, colors, labels):
    ax.scatter(x, y, s=85, color=c, edgecolor="white", linewidth=1.3,
               zorder=3)

# Reference lines
ax.axhline(V19_BASELINE, color="tab:blue", linestyle=":", linewidth=1.2, alpha=0.5,
           label=f"v19 baseline = {V19_BASELINE:.4f}")
ax.axhline(PRIOR_LEADER, color="black", linestyle="--", linewidth=1.0, alpha=0.5,
           label=f"prior LB leader = {PRIOR_LEADER:.4f} (pre-v46)")

# Headline annotation on v47_s2 (best single seed, +0.090 over v19)
v47s2_x = labels.index("v47_s2")
v47s2_y = RUNS[v47s2_x][1]
ax.annotate("v47_s2 = 0.8355\nbest seed, +0.090 vs v19",
            xy=(v47s2_x, v47s2_y),
            xytext=(v47s2_x - 6.0, v47s2_y + 0.012),
            fontsize=10, fontweight="bold", color="#0E5C2F",
            ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color="#0E5C2F",
                            lw=1.4, connectionstyle="arc3,rad=-0.2"))

# Secondary annotation on v47 ensemble (#1 on public LB)
v47_x = labels.index("v47")
v47_y = RUNS[v47_x][1]
ax.annotate("v47 ensemble = 0.8264\n#1 on LB, +0.013 vs next-best",
            xy=(v47_x, v47_y),
            xytext=(v47_x - 4.0, 0.787),
            fontsize=9, fontweight="bold", color="tab:green",
            ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color="tab:green",
                            lw=1.2, connectionstyle="arc3,rad=0.25"))

# Vertical bracket showing v47 seed range (0.0229, ~3x v46's 0.0072)
v47_seed_x = labels.index("v47_s2")
seed_top = 0.8355
seed_bot = 0.8126
ax.plot([v47_seed_x + 1.4, v47_seed_x + 1.4], [seed_bot, seed_top],
        color="#0E5C2F", lw=1.2, alpha=0.75)
ax.plot([v47_seed_x + 1.3, v47_seed_x + 1.5], [seed_top, seed_top],
        color="#0E5C2F", lw=1.2, alpha=0.75)
ax.plot([v47_seed_x + 1.3, v47_seed_x + 1.5], [seed_bot, seed_bot],
        color="#0E5C2F", lw=1.2, alpha=0.75)
ax.text(v47_seed_x + 1.7, (seed_top + seed_bot) / 2,
        "v47 seed\nrange = 0.0229\n(~3× v46's 0.0072)",
        fontsize=8, color="#0E5C2F", ha="left", va="center")

# Smaller annotations on key inflections
ax.annotate("v41 +0.011\n(L4 regularizers)",
            xy=(labels.index("v41"), 0.7563),
            xytext=(labels.index("v41") - 2.2, 0.78),
            fontsize=9, color="tab:blue", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.9))

ax.annotate("v44 +0.025\n(hard pseudo)",
            xy=(labels.index("v44"), 0.7812),
            xytext=(labels.index("v44") - 2.2, 0.81),
            fontsize=9, color="tab:green", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:green", lw=0.9))

ax.annotate("v43 −0.012 regression\n(stacked L4 changes)",
            xy=(labels.index("v43"), 0.7444),
            xytext=(labels.index("v43") - 1.5, 0.66),
            fontsize=9, color="tab:red", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.9))

ax.annotate("v42 SSL collapse\n(patient shortcut)",
            xy=(labels.index("v42"), 0.5908),
            xytext=(labels.index("v42") - 0.5, 0.555),
            fontsize=9, color="tab:red", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.9))

# v48 placeholder zone (orange) — Hinton T=2 distillation queued
v48_x = len(RUNS)
ax.axvspan(v48_x - 0.4, v48_x + 0.4, color="tab:orange", alpha=0.10, zorder=0)
ax.text(v48_x, 0.85, "v48\n(queued)", color="tab:orange",
        fontsize=10, fontweight="bold", ha="center", va="bottom")
ax.scatter(v48_x, 0.84, s=120, color="tab:orange", marker="o",
           edgecolor="white", linewidth=1.5, alpha=0.55, zorder=3)
ax.text(v48_x, 0.84, "?", color="white", fontsize=11, fontweight="bold",
        ha="center", va="center", zorder=4)

# X axis
ax.set_xticks(list(xs) + [v48_x])
ax.set_xticklabels(labels + ["v48"], rotation=45, ha="right", fontsize=9)

# Y axis
ax.set_ylabel("Public Kaggle LB (AUC)", fontsize=12)
ax.set_xlabel("Chronological version", fontsize=11)
ax.set_ylim(0.54, 0.90)
ax.set_yticks([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
ax.grid(True, axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Title
ax.set_title("Public LB progression across 30 logged iterations  ·  v19 → v47_s2 = +0.090 (best seed) / v47 ens = +0.081",
             fontsize=13, fontweight="bold", pad=12, loc="left")

# Custom legend (color categories)
legend_handles = [
    mpatches.Patch(color="tab:green", label="Best / breakthrough (ensemble)"),
    mpatches.Patch(color="tab:blue",  label="Useful gain"),
    mpatches.Patch(color="darkgreen", label="v46 per-seed extract"),
    mpatches.Patch(color="#0E5C2F",   label="v47 per-seed extract"),
    mpatches.Patch(color="tab:red",   label="Regression / negative result"),
    mpatches.Patch(color="tab:gray",  label="Neutral / replication"),
    mpatches.Patch(color="tab:orange", label="v48 (queued)"),
]
# Two-line legend setup: marker legend on left, axhline legend on right
leg1 = ax.legend(handles=legend_handles, loc="lower right",
                 frameon=True, fontsize=9, title="Outcome category",
                 title_fontsize=9, ncol=1)
ax.add_artist(leg1)
ax.legend(loc="upper left", frameon=True, fontsize=8.5)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT_PNG}")
print(f"Saved {OUT_PDF}")
print(f"Image size: {OUT_PNG.stat().st_size / 1024:.1f} KB")
