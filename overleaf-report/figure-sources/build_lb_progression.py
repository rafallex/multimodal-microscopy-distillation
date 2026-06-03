"""Build the LB-progression figure for the A3 deck — positive milestones only.

The four versions that each set a new public-LB best, from the supervised floor
(v19) to the soft-distillation breakthrough (v46). No dips / negative results.

Outputs:
  presentation/figures/lb_progression.png   (used by the deck + README)
  overleaf-report/lb_progression.pdf
"""
import matplotlib.pyplot as plt
from pathlib import Path

# (label, public LB, marker colour, delta-or-None, description)
RUNS = [
    ("v19", 0.7455, "#1A3A52", None,     "supervised floor"),
    ("v41", 0.7563, "#2563EB", "+0.011", "regularisers"),
    ("v44", 0.7812, "#059669", "+0.025", "hard pseudo · Lee 2013"),
    ("v46", 0.8236, "#D97706", "+0.042", "soft distillation · Hinton 2015"),
]
V19_BASELINE = 0.7455

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
OUT_PNG = PROJECT_ROOT / "presentation" / "figures" / "lb_progression.png"
OUT_PDF = PROJECT_ROOT / "overleaf-report" / "lb_progression.pdf"

fig, ax = plt.subplots(figsize=(13, 6.5), dpi=120)
xs = list(range(len(RUNS)))
ys = [r[1] for r in RUNS]

# Rising connecting line + milestone markers
ax.plot(xs, ys, color="#94A3B8", linewidth=2.5, zorder=1)
for x, r in zip(xs, RUNS):
    ax.scatter(x, r[1], s=240, color=r[2], edgecolor="white", linewidth=2.5, zorder=3)
    ax.annotate(f"{r[1]:.4f}", xy=(x, r[1]), xytext=(0, 17), textcoords="offset points",
                ha="center", fontsize=12.5, fontweight="bold", color=r[2])
    if r[3]:
        ax.annotate(r[3], xy=(x, r[1]), xytext=(0, -30), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold", color=r[2])

# v19 baseline reference
ax.axhline(V19_BASELINE, color="#94A3B8", linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(-0.45, V19_BASELINE + 0.0012, f"v19 baseline = {V19_BASELINE:.4f}",
        fontsize=9, color="#64748B", va="bottom")

# Headline callout on v46
ax.annotate("the breakthrough\n+0.078 over v19",
            xy=(3, ys[-1]), xytext=(2.05, ys[-1] + 0.004),
            fontsize=12, fontweight="bold", color="#D97706", ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.6,
                            connectionstyle="arc3,rad=-0.25"))

ax.set_xticks(xs)
ax.set_xticklabels([f"{r[0]}\n{r[4]}" for r in RUNS], fontsize=11)
ax.set_xlim(-0.55, 3.55)
ax.set_ylim(0.72, 0.845)
ax.set_yticks([0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84])
ax.set_ylabel("Public Kaggle LB (AUC)", fontsize=13)
ax.set_xlabel("Version (each a new best)", fontsize=11)
ax.grid(True, axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Public LB progression  ·  v19 → v46 distillation = +0.078",
             fontsize=14, fontweight="bold", pad=14, loc="left")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT_PNG}")
print(f"Image size: {OUT_PNG.stat().st_size / 1024:.1f} KB")
