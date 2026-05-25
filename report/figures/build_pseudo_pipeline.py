"""Build Fig. 2 — pseudo-label pipeline (v44 hard, v46 soft, v47 round 2).

Shows the teacher → test cells → hard/soft split → student → output chain,
with v47's iterative noisy-student round-2 step appended.

Output: report/figures/pseudo_pipeline.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "pseudo_pipeline.png"

# Palette
NAVY    = "#1A3A52"
TEAL    = "#0D9488"
AMBER   = "#D97706"
EMERALD = "#059669"
CRIMSON = "#B91C1C"
CREAM   = "#FAF7F2"
MUTED   = "#64748B"
WHITE   = "#FFFFFF"

# === Figure ===
fig, ax = plt.subplots(figsize=(12, 5.6), dpi=150)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.6)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, title, lines=None, *,
        face=WHITE, edge=TEAL, text_color=NAVY,
        title_size=10, line_size=8.5, accent_strip=True):
    bbox = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.04,rounding_size=0.10",
                          linewidth=1.2, edgecolor=edge, facecolor=face,
                          zorder=2)
    ax.add_patch(bbox)
    if accent_strip:
        strip = FancyBboxPatch((x, y), w * 0.05, h,
                               boxstyle="round,pad=0.04,rounding_size=0.10",
                               linewidth=0, facecolor=edge, zorder=3)
        ax.add_patch(strip)
    ax.text(x + w / 2 + (w * 0.025), y + h - 0.22, title,
            ha="center", va="top", fontsize=title_size, fontweight="bold",
            color=text_color, zorder=4)
    if lines:
        for i, line in enumerate(lines):
            ax.text(x + w / 2 + (w * 0.025), y + h - 0.50 - i * 0.26, line,
                    ha="center", va="top", fontsize=line_size, color=MUTED,
                    zorder=4)


def arrow(x1, y1, x2, y2, *, color=MUTED, lw=1.6, style="-|>",
          curve=None, label=None, label_xy=None, label_color=None):
    if curve is None:
        arr = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle=style, mutation_scale=16,
                              color=color, lw=lw, zorder=1)
    else:
        arr = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle=style, mutation_scale=16,
                              color=color, lw=lw, zorder=1,
                              connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(arr)
    if label:
        lx, ly = label_xy if label_xy else ((x1 + x2) / 2, (y1 + y2) / 2 + 0.18)
        ax.text(lx, ly, label, fontsize=8.5, fontweight="bold",
                color=label_color or color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor=WHITE,
                          edgecolor=color, linewidth=0.9, alpha=0.95))


# === Title ===
ax.text(0.2, 5.30,
        "Pseudo-labels & iterative noisy student · v44 (hard) → v46 (soft) → v47 (round 2)",
        fontsize=11.5, fontweight="bold", color=NAVY, ha="left", va="top")

# === Layout (left → right) ===
# Stage 1: Real training data + teacher
box(0.20, 2.95, 1.90, 1.10, "Real labels",
    ["114,302 cells", "12 patients", "patient_id ≥ 0"],
    face=CREAM, edge=NAVY)

# Stage 2: Teacher (v41 / v44_seed1)
box(0.20, 1.10, 1.90, 1.10, "Teacher",
    ["v41 hard pseudo", "v44_seed1 soft", "LB 0.756 / 0.784"],
    face=WHITE, edge=AMBER)

# Stage 3: Unlabelled test cells
box(2.60, 2.05, 1.90, 1.10, "Unlabelled",
    ["59,040 test cells", "patient_id = −1"],
    face=CREAM, edge=MUTED)

# Stage 4a: Hard split (top branch)
box(5.00, 3.60, 2.40, 1.10, "HARD pseudo",
    ["Lee 2013 · thr 0.05/0.95", "keeps tails only · 9k cells",
     "discards 84% of test cells"],
    face=WHITE, edge=CRIMSON, title_size=9.5)

# Stage 4b: Soft split (bottom branch)
box(5.00, 0.30, 2.40, 1.10, "SOFT pseudo",
    ["Hinton 2015 · raw probs", "keeps all 59,040 cells",
     "preserves teacher confidence"],
    face=WHITE, edge=EMERALD, title_size=9.5)

# Stage 5: Student (center, large)
box(7.95, 1.85, 1.85, 1.50, "Student",
    ["EffNet-B0 dual-branch", "3 seeds × SWA",
     "40-way TTA", "L = BCE(real)", "+ 0.5 · BCE(pseudo)"],
    face=CREAM, edge=NAVY, title_size=10)

# Stage 6: Results — vertical stack on right
box(10.20, 3.85, 1.65, 0.85, "v44 · +0.025",
    ["LB 0.7812 · hard"],
    face=WHITE, edge=EMERALD, title_size=11)
box(10.20, 2.20, 1.65, 0.85, "v46 · +0.039",
    ["LB 0.8236 · soft"],
    face=WHITE, edge=AMBER, title_size=11)
box(10.20, 0.55, 1.65, 0.85, "v47 · +0.003",
    ["LB 0.8264 · round 2"],
    face=WHITE, edge=TEAL, title_size=11)

# === Arrows ===
# Teacher → unlabelled
arrow(2.10, 1.65, 2.60, 2.30)
# Real labels → student (direct)
arrow(2.10, 3.50, 7.95, 2.95, curve=-0.15, color=NAVY)
# Unlabelled → hard branch
arrow(4.50, 2.85, 5.00, 4.05, color=CRIMSON, curve=0.10)
# Unlabelled → soft branch
arrow(4.50, 2.30, 5.00, 0.95, color=EMERALD, curve=-0.10)
# Hard → student
arrow(7.40, 4.15, 8.20, 3.35, color=CRIMSON)
# Soft → student
arrow(7.40, 0.85, 8.20, 1.85, color=EMERALD)
# Student → v44 result (top)
arrow(9.80, 3.25, 10.20, 4.25, color=EMERALD, curve=0.12)
# Student → v46 result (middle)
arrow(9.80, 2.60, 10.20, 2.60, color=AMBER)
# Round-2 loop: v46 result → back to teacher (round 2)
arrow(11.00, 2.20, 11.85, 1.40, color=TEAL, curve=-0.30,
      label="round 2", label_xy=(11.55, 1.85), label_color=TEAL)
arrow(11.85, 1.40, 7.95, 2.05, color=TEAL, curve=0.50, lw=1.4)
# Student → v47 result (bottom)
arrow(9.80, 1.95, 10.20, 0.95, color=TEAL, curve=-0.12)

# Bottom-left caption / formula
ax.text(0.2, 0.05,
        "L = BCE(real cells, hard labels) + 0.5 · BCE(pseudo cells, teacher probs)",
        fontsize=9.5, fontweight="bold", color=NAVY, ha="left", va="bottom")

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
plt.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.15)
print(f"Saved {OUT}")
print(f"Saved {OUT.with_suffix('.pdf')}")
print(f"PNG size: {OUT.stat().st_size / 1024:.1f} KB")
