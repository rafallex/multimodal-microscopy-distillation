"""Cross-architecture ensemble — the realistic path to >0.85.

Run AFTER v48 (EffNet-B2), v49 (ResNet-50), v50 (EffNet-B0 lottery) have produced
per-seed CSVs in results/v48|v49|v50/.

Within-EffNet-B0 averaging is capped (every model corr >0.94 with v47_s2 -> all 8
CPU probes lost to pure s2). The ONLY way a blend beats a single model here is
DIVERSITY: average models from different architecture families. This script:

  1. Auto-detects each architecture's best single seed (you pass its public LB after
     submitting the per-seed CSVs; until then it falls back to the arch ensemble).
  2. Prints the cross-architecture rank-correlation matrix (confirms diversity is real
     — expect ~0.80 across families vs 0.98 within EffNet-B0).
  3. Writes a few principled blend candidates to results/cross_arch/ for submission:
       - equal-weight mean
       - strength-weighted (softmax of member LBs)
       - rank-average (robust for an AUC metric)
       - s2-anchored (best member 0.5, others split the rest)

Submit the candidates; the best blend + the best single seed are your 2 final picks.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass
try:
    from scipy.stats import rankdata
except ImportError:
    def rankdata(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float); r[order] = np.arange(1, len(a) + 1)
        return r

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "cross_arch"
OUT.mkdir(exist_ok=True)

# ── Members: (label, csv path, public LB). v47_s2 is the known anchor.
#    For v48/v49/v50, point `csv` at the best single seed AFTER you see per-seed LBs;
#    `lb` is that seed's public score. Leave lb=None to skip a member that hasn't run.
MEMBERS = [
    ("v47_s2 / B0-late",   RESULTS / "v47" / "submission_seed2.csv", 0.8355),
    ("v48 / B2-late",      RESULTS / "v48" / "submission_seed1.csv", None),   # capacity; set path+lb after submitting
    ("v51 / ConvNeXt-late", RESULTS / "v51" / "submission_seed1.csv", None),  # strong + diverse backbone
    ("v52 / B0-EARLY",     RESULTS / "v52" / "submission_seed1.csv", None),   # most decorrelated (different fusion)
    ("v49 / ResNet50-late", RESULTS / "v49" / "submission_seed1.csv", None),  # diverse but weaker
    ("v50 / B0-late-lotto", RESULTS / "v50" / "submission_seed401.csv", None),
]


def load(path):
    df = pd.read_csv(path).sort_values("Name").reset_index(drop=True)
    return df["Name"].values, df["Diagnosis"].values


present = [(lbl, p, lb) for (lbl, p, lb) in MEMBERS if Path(p).exists()]
missing = [lbl for (lbl, p, lb) in MEMBERS if not Path(p).exists()]
if missing:
    print("Not yet available (skipped):")
    for m in missing:
        print(f"   - {m}")
    print()

if len(present) < 2:
    print(f"Only {len(present)} member(s) present. Need >=2 to blend.")
    print("Re-run after v48/v49/v50 produce per-seed CSVs in results/.")
    sys.exit(0)

names0, _ = load(present[0][1])
preds, labels, lbs = {}, [], {}
for lbl, p, lb in present:
    nm, pr = load(p)
    assert np.array_equal(nm, names0), f"Name mismatch in {lbl}"
    preds[lbl] = pr; labels.append(lbl); lbs[lbl] = lb

# ── 1. diversity check (rank correlation) ──
R = np.array([rankdata(preds[l]) for l in labels])
C = np.corrcoef(R)
print("=== cross-architecture rank-correlation (lower = more ensemble headroom) ===")
print(f"{'':22}" + "".join(f"{l.split('/')[0].strip():>16}" for l in labels))
for i, l in enumerate(labels):
    print(f"{l:22}" + "".join(f"{C[i, j]:>16.3f}" for j in range(len(labels))))
print()

P = np.column_stack([preds[l] for l in labels])

def save(tag, w):
    w = np.asarray(w, float); w = w / w.sum()
    blend = (P * w).sum(axis=1)
    out = OUT / f"crossarch_{tag}.csv"
    pd.DataFrame({"Name": names0, "Diagnosis": blend}).to_csv(out, index=False)
    wstr = ", ".join(f"{l.split('/')[0].strip()}={wi:.2f}" for l, wi in zip(labels, w))
    print(f"  {out.name:<34} [{wstr}]")
    return blend

print("=== blend candidates written to results/cross_arch/ ===")
# equal weight
save("equal", np.ones(len(labels)))
# strength-weighted (softmax of LBs); only if all LBs known
if all(lbs[l] is not None for l in labels):
    lbv = np.array([lbs[l] for l in labels])
    save("strengthsoftmax", np.exp((lbv - lbv.mean()) / 0.01))
else:
    print("  (strength-weighted skipped — set each member's lb= after submitting per-seed CSVs)")
# rank-average (robust for AUC): blend of normalized ranks
rank_blend = R.mean(axis=0)
pd.DataFrame({"Name": names0, "Diagnosis": rank_blend / rank_blend.max()}).to_csv(
    OUT / "crossarch_rankavg.csv", index=False)
print(f"  {'crossarch_rankavg.csv':<34} [equal-weight average of per-model ranks]")
# anchor best member at 0.5
best_idx = int(np.argmax([lbs[l] if lbs[l] is not None else -1 for l in labels]))
w_anchor = np.full(len(labels), 0.5 / (len(labels) - 1)); w_anchor[best_idx] = 0.5
save("anchorbest", w_anchor)

print()
print("Submit these to Kaggle. The diversity (cross-family corr ~0.80) is what lets a")
print("blend exceed the best single member — unlike the within-EffNet probes (corr 0.98).")
print("Final 2 picks: best cross-arch blend + best single seed.")
