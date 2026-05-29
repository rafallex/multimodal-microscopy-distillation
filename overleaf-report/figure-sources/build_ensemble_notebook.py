"""Local generator -> emits notebooks/ensemble_crossarch_source.ipynb (a KAGGLE notebook).

You run the .ipynb on Kaggle; this .py just builds it locally. The ensemble notebook
is CPU-only: it reads the per-model submission CSVs you attach as Kaggle inputs and
writes blended submission candidates to /kaggle/working.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "notebooks" / "ensemble_crossarch_source.ipynb"

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}

cells = []

cells.append(md("""# Cross-Architecture Ensemble (CPU Kaggle notebook) — blend diverse models into one submission

Run this AFTER the training notebooks (v48 B2 / v49 ResNet-50 / v51 ConvNeXt / v52 early-fusion)
have produced per-seed `submission_seed{N}.csv`, plus the v47_s2 anchor.

**Why blend:** within-EffNet-B0 averaging is capped (all our B0 models corr >0.94, so every
average loses to the best single seed). This blends models from DIFFERENT architecture/fusion
families (corr ~0.72-0.80) — the only configuration where the average can exceed any single
member, i.e. the path to >0.85.

## How to run on Kaggle
1. **Settings -> Accelerator = None** (CPU is enough; saves your GPU quota).
2. **Add Input** for each model you want to blend: attach each training notebook's *Output*
   (and the `submissionv47seed2` dataset for the v47_s2 anchor). Attach only the CSVs you
   want in the blend.
3. **Run All.** It auto-discovers every `submission*.csv` under `/kaggle/input`, prints the
   rank-correlation matrix (diversity check), and writes blended candidates to `/kaggle/working`.
4. Submit the `crossarch_*.csv` files. **Final 2 picks = best blend + best single seed.**
"""))

cells.append(code('''import os, glob
import numpy as np, pandas as pd
try:
    from scipy.stats import rankdata
except Exception:
    def rankdata(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float); r[order] = np.arange(1, len(a) + 1)
        return r

# Substring of the COMPETITION data folder, so we never blend train/sample files:
COMP_HINT = "multimodal-cancer"
# Auto-discover every submission*.csv you attached (recommended). Set False to use MEMBERS.
AUTODISCOVER = True
# Explicit members if AUTODISCOVER=False: (label, path, weight-or-None)
MEMBERS = [
    ("v47_s2", "/kaggle/input/submissionv47seed2/submission_seed2.csv", None),
    # ("v48_b2", "/kaggle/input/<v48-output>/submission_seed1.csv", None),
]
OUT_DIR = "/kaggle/working"
'''))

cells.append(code('''# ---- discover + load ----
def discover():
    found = []
    for p in glob.glob("/kaggle/input/**/submission*.csv", recursive=True):
        if COMP_HINT in p.lower():        # skip competition sampleSubmission
            continue
        label = os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)
        found.append((label, p, None))
    return sorted(found)

members = discover() if AUTODISCOVER else MEMBERS
members = [(l, p, w) for (l, p, w) in members if os.path.exists(p)]
print(f"Found {len(members)} submission CSV(s) to blend:")
for l, p, w in members:
    print(f"   {l:42}  {p}")
assert len(members) >= 2, "Attach at least 2 model CSVs (Add Input -> notebook outputs)."

names0 = None
preds, labels, weights = {}, [], {}
for l, p, w in members:
    df = pd.read_csv(p)
    if not {"Name", "Diagnosis"} <= set(df.columns):
        print(f"   skip {l}: not a (Name,Diagnosis) submission"); continue
    df = df.sort_values("Name").reset_index(drop=True)
    if names0 is None:
        names0 = df["Name"].values
    assert np.array_equal(df["Name"].values, names0), f"Name mismatch: {l}"
    preds[l] = df["Diagnosis"].values.astype(float); labels.append(l); weights[l] = w
print(f"\\nLoaded {len(labels)} aligned predictors over {len(names0)} cells.")
'''))

cells.append(code('''# ---- diversity check: rank correlation ----
R = np.array([rankdata(preds[l]) for l in labels])
C = np.corrcoef(R)
print("Rank-correlation matrix (lower off-diagonal = more ensemble headroom):")
print(f"{'':24}" + "".join(f"{l[:14]:>16}" for l in labels))
for i, l in enumerate(labels):
    print(f"{l[:24]:24}" + "".join(f"{C[i, j]:>16.3f}" for j in range(len(labels))))
'''))

cells.append(code('''# ---- build + write blend candidates ----
P = np.column_stack([preds[l] for l in labels])

def write(tag, vec):
    out = os.path.join(OUT_DIR, f"crossarch_{tag}.csv")
    pd.DataFrame({"Name": names0, "Diagnosis": vec}).to_csv(out, index=False)
    print(f"  wrote {out}  mean={vec.mean():.4f}")

# 1) equal-weight probability mean
write("equal", P.mean(axis=1))

# 2) rank-average (robust for an AUC metric): mean of per-model ranks, normalized
rank_blend = R.mean(axis=0); write("rankavg", rank_blend / rank_blend.max())

# 3) custom-weighted, if you set weights in MEMBERS (else skipped)
if all(weights[l] is not None for l in labels):
    w = np.array([weights[l] for l in labels], float); w = w / w.sum()
    write("weighted", (P * w).sum(axis=1))
else:
    print("  (weighted blend skipped — set weights in MEMBERS to enable)")

# 4) anchor-heavy: if 'v47_s2' present, give it 0.5 and split the rest equally
anchor = next((i for i, l in enumerate(labels) if "v47_s2" in l or "s2" in l.lower()), None)
if anchor is not None and len(labels) >= 2:
    w = np.full(len(labels), 0.5 / (len(labels) - 1)); w[anchor] = 0.5
    write("anchor_s2", (P * w).sum(axis=1))

print("\\nSubmit the crossarch_*.csv files. Best blend + best single seed = your 2 final picks.")
'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# validate
import nbformat
nbformat.validate(nbformat.read(str(OUT), as_version=4))
print(f"Saved + VALIDATED {OUT.name} ({len(cells)} cells) — CPU Kaggle ensemble notebook.")
