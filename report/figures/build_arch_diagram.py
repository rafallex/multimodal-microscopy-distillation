"""Build Fig. 1 — dual EfficientNet-B0 backbone diagram with MIL aux loss.

Shows: BF + FL inputs → two EffNet-B0 backbones → GAP → concat → MLP head → ŷ.
Side branch shows the per-patient MIL aux loss.

Output: report/figures/arch_diagram.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ConnectionPatch

OUT = Path(__file__).resolve().parent / "arch_diagram.png"

# Palette consistent with the deck
NAVY    = "#1A3A52"
TEAL    = "#0D9488"
AMBER   = "#D97706"
EMERALD = "#059669"
CREAM   = "#FAF7F2"
MUTED   = "#64748B"
WHITE   = "#FFFFFF"
SLATE   = "#334155"
LIGHT   = "#E5E7EB"

# === Figure ===
fig, ax = plt.subplots(figsize=(12, 5.0), dpi=150)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, title, lines=None, *,
        face=WHITE, edge=TEAL, text_color=NAVY, title_size=10,
        line_size=8.5, accent_strip=True):
    """Draw a labeled rounded box with optional sub-lines."""
    bbox = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.04,rounding_size=0.10",
                         linewidth=1.2, edgecolor=edge, facecolor=face,
                         zorder=2)
    ax.add_patch(bbox)
    if accent_strip:
        strip = FancyBboxPatch((x, y), w * 0.06, h,
                               boxstyle="round,pad=0.04,rounding_size=0.10",
                               linewidth=0, facecolor=edge, zorder=3)
        ax.add_patch(strip)
    # Title near top of box
    ax.text(x + w / 2 + (w * 0.03), y + h - 0.22, title,
            ha="center", va="top", fontsize=title_size, fontweight="bold",
            color=text_color, zorder=4)
    # Sub-lines below
    if lines:
        for i, line in enumerate(lines):
            ax.text(x + w / 2 + (w * 0.03), y + h - 0.50 - i * 0.26, line,
                    ha="center", va="top", fontsize=line_size, color=MUTED,
                    zorder=4)


def arrow(x1, y1, x2, y2, *, color=MUTED, lw=1.8):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle="-|>", mutation_scale=18,
                          color=color, lw=lw, zorder=1)
    ax.add_patch(arr)


# Layout: left → right horizontal flow at two y-levels (BF top, FL bottom)
# Inputs
box(0.20, 3.30, 1.50, 1.10, "Brightfield",
    ["128 × 128 × 1", "grayscale"], face=CREAM, edge=NAVY)
box(0.20, 0.60, 1.50, 1.10, "Fluorescence",
    ["128 × 128 × 1", "grayscale"], face=CREAM, edge=NAVY)

# Backbones
box(2.30, 3.10, 2.50, 1.50, "EfficientNet-B0",
    ["timm · ra_in1k", "grayscale conv_stem", "≈ 4M params"],
    face=WHITE, edge=TEAL)
box(2.30, 0.40, 2.50, 1.50, "EfficientNet-B0",
    ["timm · ra_in1k", "grayscale conv_stem", "≈ 4M params"],
    face=WHITE, edge=TEAL)

# GAP (small boxes)
box(5.30, 3.50, 1.00, 0.70, "GAP",
    ["1 × 1280"], face=CREAM, edge=MUTED, title_size=9, line_size=7.5)
box(5.30, 0.80, 1.00, 0.70, "GAP",
    ["1 × 1280"], face=CREAM, edge=MUTED, title_size=9, line_size=7.5)

# Concat
box(6.80, 2.05, 1.60, 0.90, "concat",
    ["2 × 1280", "late fusion"], face=WHITE, edge=AMBER)

# MLP head
box(9.00, 2.05, 2.00, 0.90, "MLP head",
    ["512 → 1", "logit · σ → ŷ"], face=WHITE, edge=NAVY)

# Output indicator
ax.text(11.40, 2.50, "ŷ", fontsize=18, fontweight="bold",
        color=AMBER, ha="center", va="center")

# MIL aux box (bottom)
box(7.20, 0.00, 3.80, 0.95, "MIL aux loss",
    ["mean cell logit per patient → BCE  (weight 0.5)"],
    face=CREAM, edge=EMERALD, title_size=9.5, line_size=8.5)

# === Arrows ===
# Inputs → backbones
arrow(1.70, 3.85, 2.30, 3.85)
arrow(1.70, 1.15, 2.30, 1.15)
# Backbones → GAP
arrow(4.80, 3.85, 5.30, 3.85)
arrow(4.80, 1.15, 5.30, 1.15)
# GAP → concat
arrow(6.30, 3.85, 6.95, 2.95)
arrow(6.30, 1.15, 6.95, 2.05)
# concat → MLP
arrow(8.40, 2.50, 9.00, 2.50)
# MLP → ŷ
arrow(11.00, 2.50, 11.30, 2.50, color=AMBER)
# MLP → MIL aux  (curved dashed)
mil_arrow = FancyArrowPatch((9.50, 2.05), (9.10, 0.95),
                             arrowstyle="-|>", mutation_scale=14,
                             color=EMERALD, lw=1.4,
                             connectionstyle="arc3,rad=-0.25", zorder=1,
                             linestyle="--")
ax.add_patch(mil_arrow)

# Title
ax.text(0.20, 4.78, "Dual EfficientNet-B0 backbone · late concat fusion · per-patient MIL aux",
        fontsize=11.5, fontweight="bold", color=NAVY, ha="left", va="top")

# Save
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
plt.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.15)
print(f"Saved {OUT}")
print(f"Saved {OUT.with_suffix('.pdf')}")
print(f"PNG size: {OUT.stat().st_size / 1024:.1f} KB")
