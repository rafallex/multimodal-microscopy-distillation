"""Build the LB-progression figure for the A3 report and presentation.

Plots public LB vs chronological version order, annotates the key milestones,
and adds horizontal reference lines for the v19 baseline and the prior leader.

Output: results/figures/lb_progression.png (300 dpi, ~1600x900 px)
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
]
# v47 placeholder (queued)
V47_PLACEHOLDER = ("v47", None, "tab:orange")

PRIOR_LEADER = 0.7916  # was the public LB top before v46 took #1
V19_BASELINE = 0.7455

OUT = Path(__file__).resolve().parent / "lb_progression.png"

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

# Headline annotation on v46
v46_x = labels.index("v46")
v46_y = RUNS[v46_x][1]
ax.annotate("NEW BEST  #1 on LB\nv46 = 0.8236",
            xy=(v46_x, v46_y),
            xytext=(v46_x - 4.5, v46_y + 0.025),
            fontsize=11, fontweight="bold", color="tab:green",
            ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color="tab:green",
                            lw=1.4, connectionstyle="arc3,rad=-0.2"))

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

# v47 placeholder zone (orange dashed)
v47_x = len(RUNS)
ax.axvspan(v47_x - 0.4, v47_x + 0.4, color="tab:orange", alpha=0.10, zorder=0)
ax.text(v47_x, 0.83, "v47\n(queued)", color="tab:orange",
        fontsize=10, fontweight="bold", ha="center", va="bottom")
ax.scatter(v47_x, 0.82, s=120, color="tab:orange", marker="o",
           edgecolor="white", linewidth=1.5, alpha=0.55, zorder=3)
ax.text(v47_x, 0.82, "?", color="white", fontsize=11, fontweight="bold",
        ha="center", va="center", zorder=4)

# X axis
ax.set_xticks(list(xs) + [v47_x])
ax.set_xticklabels(labels + ["v47"], rotation=45, ha="right", fontsize=9)

# Y axis
ax.set_ylabel("Public Kaggle LB (AUC)", fontsize=12)
ax.set_xlabel("Chronological version", fontsize=11)
ax.set_ylim(0.54, 0.88)
ax.set_yticks([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
ax.grid(True, axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Title
ax.set_title("Public LB progression across 29 logged iterations  ·  v19 → v46 = +0.078",
             fontsize=14, fontweight="bold", pad=12, loc="left")

# Custom legend (color categories)
legend_handles = [
    mpatches.Patch(color="tab:green", label="Best / breakthrough"),
    mpatches.Patch(color="tab:blue",  label="Useful gain"),
    mpatches.Patch(color="darkgreen", label="v46 per-seed extract"),
    mpatches.Patch(color="tab:red",   label="Regression / negative result"),
    mpatches.Patch(color="tab:gray",  label="Neutral / replication"),
    mpatches.Patch(color="tab:orange", label="v47 (queued)"),
]
# Two-line legend setup: marker legend on left, axhline legend on right
leg1 = ax.legend(handles=legend_handles, loc="lower right",
                 frameon=True, fontsize=9, title="Outcome category",
                 title_fontsize=9, ncol=1)
ax.add_artist(leg1)
ax.legend(loc="upper left", frameon=True, fontsize=8.5)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print(f"Saved {OUT}")
print(f"Saved {OUT.with_suffix('.pdf')}")
print(f"Image size: {OUT.stat().st_size / 1024:.1f} KB")
