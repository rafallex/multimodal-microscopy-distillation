"""Build notebooks/improvedv49_source.ipynb by copying v47 and patching 2 cells.

v49 = v47 recipe with ONE knob changed: teacher CSV swapped from v46 ensemble
(LB 0.8236) to v47_seed2 (LB 0.8355). Tests whether the 97%-middle-band
finding (v47_s2's lift is concentrated in the teacher's uncertain region)
compounds when v47_s2 itself becomes the teacher for round 3.

Single-knob discipline: ONLY the teacher CSV path changes. All hyperparameters
(seeds, SWA, TTA, weight decay, pseudo loss weight, augmentation, label
smoothing, dropout) are identical to v47.

Run this script to (re)generate the notebook from the v47 source.
"""
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V47_NB = PROJECT_ROOT / "notebooks" / "improvedv47_source.ipynb"
V49_NB = PROJECT_ROOT / "notebooks" / "improvedv49_source.ipynb"

with open(V47_NB, encoding="utf-8") as f:
    nb = json.load(f)

# === Patch cell 0 (markdown header) ===
new_md = """# Multimodal Cancer Classification Challenge 2026 — v49 (noisy student round 3 with best-seed teacher)

**v49 = same recipe as v47, but with v47_seed2 as the new teacher.** Third iteration of the noisy-student loop (Xie et al. 2020). v47_seed2 reached LB 0.8355 — the highest single-seed score in the project, and 0.0119 higher than v46 ensemble (LB 0.8236) which served as the v47 teacher.

## What v49 changes vs v47

| Component | v47 | **v49 (this)** |
|---|---|---|
| Pseudo teacher | v46 ensemble (LB 0.8236) | **v47_seed2 (LB 0.8355)** — best single seed of round 2 |
| Everything else | — | **identical** (soft pseudo, weight 0.5, no FL aug, WD 1e-4, 3 seeds × SWA, 40-way TTA, label smoothing 0.05, dropout 0.4) |

That's the only diff. Single-knob test of "does a better teacher (higher-LB, single-seed) compound the dark-knowledge signal?"

## Why this is the highest-EV play we haven't run

Paper §VIII-E (noise-floor analysis) found that v46 → v47 ensemble lift (+0.003) is **indistinguishable from zero** (0.27σ vs the per-seed SE = 0.0073). That seemed to imply round 3 wouldn't help either.

BUT a per-cell analysis of v47_s2 vs v47 ensemble (paper §VIII-E second paragraph) found:

**97% of |p_s2 − p_ensemble| is concentrated in cells where the v46 teacher predicted p ∈ [0.05, 0.95]** — the "dark knowledge" middle band. By ensemble-confidence decile:
- D1–D5 and D8–D10 (confident-class cells): 0–1% sign-flip rate
- D6 (p_ens 0.46–0.59): **58% sign-flip rate**
- D7 (p_ens 0.59–0.72): **33% sign-flip rate**

s2 and the ensemble agree on every confident class call and diverge sharply on exactly the cells where there's genuine ambiguity. This is the signature of a model that found a better decision boundary in the dark-knowledge region — not a model that got lucky on the public split.

**If s2's middle-band calibration is signal**, training a round-3 student against s2's targets should propagate that signal forward. The teacher delta v46_ens → v47_s2 is +0.0119 LB — **4× larger than the v46_ens → v47_ens teacher delta that gave +0.003**. If teacher quality is the bottleneck (and §VIII-C's diminishing-returns argument hinges on the teacher delta being small), then a much-better teacher should give a much-better student.

## Honest forecast

| Run | LB |
|---|---|
| v44_seed1 (round 0 teacher) | 0.7844 |
| v46 (round 1) | 0.8236 |
| v47 ensemble (round 2 with v46 teacher) | 0.8264 (+0.003) |
| v47_s2 (best single seed of round 2) | 0.8355 |
| **v49 expected range** | **0.83 — 0.86** |

- P(v49 ensemble > 0.8355): ~35–45% (genuine shot at clearing v47_s2)
- P(v49 best seed > 0.84): ~30–40%
- P(v49 best seed > 0.85): ~15–25%
- P(v49 collapses below 0.825): ~10–15% (teacher noise from using a single seed)

**The risk to flag**: v47_s2 is a single seed (SE ≈ 0.0123 per seed). Using it as a teacher inherits whatever public-split-specific patterns made it lucky. If the 97% middle-band signal is real, this compounds the win; if it's split-luck, this compounds the loss. The private-LB will adjudicate.

## Required Kaggle inputs

Attach both:
1. `rafaelproena/a3-adl` (competition data, same as always)
2. **`rafaelproena/submissionv47seed2`** — **upload `results/v47/submission_seed2.csv` (the 0.8355 single seed) as a new private Kaggle dataset named `submissionv47seed2`** (one-time setup).

Path candidates tried in order:

```
/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv
/kaggle/input/submissionv47seed2/submission_seed2.csv
/kaggle/input/v47seed2-predictions/submission_seed2.csv
```

## Compute on T4

Same as v47: ~5.5–6h. Identical training set size (114k real + 59k pseudo = 173k cells/epoch).

## Sanity check before the 5h commit

When cell 5 finishes, log should show:

```
Pseudo-labels loaded from .../submissionv47seed2/submission_seed2.csv  [mode: SOFT]
  no band filter — all cells kept
  kept 59040 cells (... >0.5, ... <=0.5) of 59040 total
  mean soft target: 0.4205 (train was 0.3876)
  Combined training set: 173342 cells (114302 real + 59040 pseudo)
```

The `mean soft target: 0.4205` is the diagnostic that we're on v47_s2, not v46 ensemble (which had mean ~0.52). If the mean is different, **abort before the 5h training runs** — wrong CSV attached.

## Strategic note

This is the single highest-EV experiment we haven't run yet. It's a one-line change from v47 and tests a precise hypothesis the per-cell analysis generated. Worth one of our 5 GPU runs in the May 30 → June 3 window.

If v49 ensemble beats 0.8355 (v47_s2): NEW BEST, take #3 → likely #1 or #2 depending on opponents.
If v49 ensemble lands 0.825–0.835: noisy-student saturated; v47_s2 stays primary pick.
If v49 ensemble < 0.825: teacher noise hypothesis confirmed; finalize v47_s2 + v41 picks.
"""

# Cell 0 is markdown
assert nb["cells"][0]["cell_type"] == "markdown", "expected cell 0 to be markdown header"
nb["cells"][0]["source"] = new_md.splitlines(keepends=True)

# === Patch cell 2 (code config — change the PSEUDO_LABEL_CANDIDATES) ===
assert nb["cells"][2]["cell_type"] == "code", "expected cell 2 to be code config"
old_src = "".join(nb["cells"][2]["source"])

# Single-knob change: swap the v46 teacher path to v47_s2
patches = [
    # Path candidates: point at v47_seed2 dataset
    (
        '"/kaggle/input/datasets/rafaelproena/submissionv46/submission.csv"',
        '"/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv"',
    ),
    (
        '"/kaggle/input/submissionv46/submission.csv"',
        '"/kaggle/input/submissionv47seed2/submission_seed2.csv"',
    ),
    (
        '"/kaggle/input/v46-predictions/submission.csv"',
        '"/kaggle/input/v47seed2-predictions/submission_seed2.csv"',
    ),
    # Update the section header comment to reflect round 3 / v47_s2 teacher
    (
        "# === v47: noisy student ROUND 2 — soft pseudo-labels from v46 (LB 0.8236) ===\n# v46 reached LB 0.8236 (current #1). Its predictions are a much stronger teacher\n# than v44_seed1 was (LB 0.7844). Per Xie 2020, iterative noisy student gains\n# accuracy across rounds. This is round 2.",
        "# === v49: noisy student ROUND 3 — soft pseudo-labels from v47_seed2 (LB 0.8355) ===\n# v47_seed2 is the best single seed of v47 (LB 0.8355, +0.0119 above v46 ensemble).\n# Per-cell analysis showed 97% of s2's disagreement with the v47 ensemble is in the\n# teacher's dark-knowledge middle band — likely-signal, not split-luck. v49 tests\n# whether that signal compounds when s2 itself becomes the round-3 teacher.",
    ),
    # Update the final print so logs are self-identifying
    (
        'print(f"\\nConfig (v47 - noisy student ROUND 2: soft pseudo from v46 / Hinton distillation):")',
        'print(f"\\nConfig (v49 - noisy student ROUND 3: soft pseudo from v47_seed2 / Hinton distillation):")',
    ),
    (
        '  USE_FL_TUNED_AUG     = {USE_FL_TUNED_AUG}  (v46 stripped; kept off in v47)',
        '  USE_FL_TUNED_AUG     = {USE_FL_TUNED_AUG}  (v46 stripped; kept off in v47/v49)',
    ),
    (
        '  WEIGHT_DECAY         = {WEIGHT_DECAY}  (v46 stripped; kept at 1e-4)',
        '  WEIGHT_DECAY         = {WEIGHT_DECAY}  (v46 stripped; kept at 1e-4 in v47/v49)',
    ),
    (
        '  USE_SOFT_PSEUDO      = {USE_SOFT_PSEUDO}  (Hinton distillation, same as v46)',
        '  USE_SOFT_PSEUDO      = {USE_SOFT_PSEUDO}  (Hinton distillation, same as v46/v47)',
    ),
]

new_src = old_src
patches_applied = 0
for old, new in patches:
    if old in new_src:
        new_src = new_src.replace(old, new)
        patches_applied += 1
    else:
        print(f"WARNING: patch did not match (skipping):\n  {old[:80]}...")

nb["cells"][2]["source"] = new_src.splitlines(keepends=True)
print(f"Patched cell 2: {patches_applied} of {len(patches)} substitutions applied")

# Save
V49_NB.parent.mkdir(parents=True, exist_ok=True)
with open(V49_NB, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nSaved {V49_NB}")
print(f"Size: {V49_NB.stat().st_size:,} bytes")

# Sanity: re-load and confirm valid JSON
with open(V49_NB, encoding="utf-8") as f:
    nb2 = json.load(f)
print(f"Cells: {len(nb2['cells'])}")
assert "v47_seed2" in "".join(nb2["cells"][0]["source"])
assert "submissionv47seed2" in "".join(nb2["cells"][2]["source"])
print("Sanity checks passed: v49 references v47_seed2 teacher in markdown + code config.")
