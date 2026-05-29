"""v52 = EARLY-FUSION EfficientNet-B0 + soft pseudo from v47_s2.

The results-folder diversity analysis found EARLY FUSION (v38: stack BF+FL as a
2-channel input to ONE backbone) is the MOST decorrelated approach we have
(rank-corr 0.717 with v47_s2, vs ResNet-50's 0.795 and within-EffNet's >0.94).
It was weak (0.7147) only because it was supervised-only. With the v47_s2
soft-pseudo lift it should reach ~0.77-0.79 AND stay maximally diverse -> the
single best new ensemble member for breaking 0.85.

Construction: take the proven v47 notebook (pseudo machinery, data pipeline,
training, TTA all unchanged) and swap ONLY the model cell (cell 4) for v38's
early-fusion model. v38's forward(bf, fl) signature matches v47's, so the data
loader, soft-pseudo targets, MIL loss, SWA and inference all work unchanged.

Bonus: early fusion uses ONE backbone (not two) -> ~half the params of v47's
dual-branch -> faster + lower memory, so batch 128 is safe.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
V47 = ROOT / "notebooks" / "improvedv47_source.ipynb"
V38 = ROOT / "results" / "v38" / "improvedv38.ipynb"
OUT = ROOT / "notebooks" / "improvedv52_source.ipynb"

nb = json.loads(V47.read_text(encoding="utf-8"))
v38_cell4 = "".join(json.loads(V38.read_text(encoding="utf-8"))["cells"][4]["source"])

# ── cell 0: markdown header ──
nb["cells"][0]["source"] = ("""# Multimodal Cancer Challenge 2026 — v52: EARLY-FUSION EffNet-B0 + soft pseudo (v47_s2 teacher)

**v52 = v47's soft-pseudo recipe, but with EARLY FUSION (BF+FL stacked as a
2-channel input to ONE backbone) instead of the dual-branch late fusion.**

## Why (results-folder diversity analysis)

Early fusion (the v38 approach) is the **most decorrelated** model we have —
rank-corr **0.717** with v47_s2, vs ResNet-50 0.795 and within-EffNet-B0 >0.94.
It processes the two modalities through a shared backbone from pixel 1, so it makes
genuinely different errors. It scored only 0.7147 supervised-only; with the v47_s2
soft-pseudo lift (+~0.06, as EffNet-B0 got from v41->v46) it should reach ~0.77-0.79
**and stay maximally diverse** — the ideal third/fourth leg of the cross-arch blend.

## What v52 changes vs v47

| Component | v47 | **v52** |
|---|---|---|
| Fusion | dual-branch late concat (2 backbones) | **early fusion: cat([BF,FL]) -> 1 backbone** |
| Backbone | EfficientNet-B0 x2 | **EfficientNet-B0 x1, 2-channel stem** (~half params) |
| Pseudo teacher | v46 ens (0.8236) | **v47_seed2 (0.8355)** |
| Seeds | [1,2,3] | **[1,2,3]** |
| Everything else | — | identical (soft pseudo all 59k, MIL, AdaBN, SWA, 40-way TTA, batch 128) |

## Required Kaggle input
`rafaelproena/a3-adl` + **`rafaelproena/submissionv47seed2`** (same dataset as v48-v51).

## Sanity check before the ~5-6 h commit (cell 4 self-test, seconds)
```
Output shape: torch.Size([2])
Backbone: EfficientNet-B0 (early-fusion, 2-ch input)
```
Teacher log must show `submissionv47seed2/...`, mean soft target ~0.42.

## Outputs / use
Per-seed `submission_seed{N}.csv` (incremental). Feed the best seed into
`build_cross_arch_ensemble.py` as the early-fusion member. Target blend:
v47_s2 (B0 late) + v48 (B2 late) + v51 (ConvNeXt late) + **v52 (B0 EARLY)** —
varying BOTH backbone and fusion = the most diverse fleet we can assemble.

## Compute
~5-6 h on T4 x2 (early fusion is ~half the params of dual-branch -> as fast or
faster than v47, batch 128 fine).
""").splitlines(keepends=True)

# ── cell 2: teacher + seeds + label (batch stays 128: early fusion is light) ──
src = "".join(nb["cells"][2]["source"])
for old, new in [
    ('"/kaggle/input/datasets/rafaelproena/submissionv46/submission.csv"',
     '"/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv"'),
    ('"/kaggle/input/submissionv46/submission.csv"',
     '"/kaggle/input/submissionv47seed2/submission_seed2.csv"'),
    ('"/kaggle/input/v46-predictions/submission.csv"',
     '"/kaggle/input/v47seed2-predictions/submission_seed2.csv"'),
    ("SEEDS               = [1, 2, 3]                  # multi-seed ensemble",
     "SEEDS               = [1, 2, 3]                  # v52: early-fusion seeds"),
    ('print(f"\\nConfig (v47 - noisy student ROUND 2: soft pseudo from v46 / Hinton distillation):")',
     'print(f"\\nConfig (v52 - EARLY-FUSION EffNet-B0 + soft pseudo from v47_seed2):")'),
]:
    assert old in src, f"cell2: {old[:45]}"
    src = src.replace(old, new)
nb["cells"][2]["source"] = src.splitlines(keepends=True)

# ── cell 4: REPLACE with v38 early-fusion model ──
nb["cells"][4]["source"] = v38_cell4.splitlines(keepends=True)

# ── cell 5: fix fallback-warning name ──
src5 = "".join(nb["cells"][5]["source"]).replace(
    "print(f\"  Attach the 'submissionv46' Kaggle dataset to enable pseudo-labels.\")",
    "print(f\"  Attach the 'submissionv47seed2' Kaggle dataset to enable pseudo-labels.\")")
nb["cells"][5]["source"] = src5.splitlines(keepends=True)

# ── cell 7: fix log string ──
src7 = "".join(nb["cells"][7]["source"]).replace(
    'print(f"  soft pseudo loss weight = {PSEUDO_LOSS_WEIGHT}  (Hinton 2015 distillation, teacher = v46)")',
    'print(f"  soft pseudo loss weight = {PSEUDO_LOSS_WEIGHT}  (teacher = v47_seed2, EARLY-FUSION EffNet-B0)")')
nb["cells"][7]["source"] = src7.splitlines(keepends=True)

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# verify
re_nb = json.loads(OUT.read_text(encoding="utf-8"))
c2 = "".join(re_nb["cells"][2]["source"]); c4 = "".join(re_nb["cells"][4]["source"])
assert "submissionv47seed2/submission_seed2.csv" in c2 and "submissionv46" not in c2
assert "torch.cat([bf, fl], dim=1)" in c4 and "in_channels=2" in c4
assert "submissionv47seed2/submission_seed2.csv" in c2
print(f"Saved {OUT.name} ({len(re_nb['cells'])} cells) — early-fusion EffNet-B0, teacher v47_s2, verified.")
