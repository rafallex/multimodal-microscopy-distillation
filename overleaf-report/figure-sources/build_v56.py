"""Builder for v56 = L1 + L2 backup (best shot at a NEW high single model).

Base: improvedv50 (B0 dual late-fusion -- the proven v47 architecture).
Changes (all value-only edits to already-tested code paths; no new logic):
  L1  teacher  -> v47 3-seed ENSEMBLE (smoother soft targets), seed2 as fallback
  L2  noise    -> RandomErasing 0.25->0.35, ColorJitter 0.4->0.5, Dropout 0.4->0.45
  seeds        -> [1, 2]

NOT run on Kaggle. Emits notebooks/improvedv56_source.ipynb. Untested on GPU by design
(no local GPU) -- the edits are param bumps to v50's proven, smoke-tested pipeline.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "notebooks" / "improvedv50_sourceDO_NOT_RUN_.ipynb"
OUT = ROOT / "notebooks" / "improvedv56_source.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))
n_edits = 0


def edit_cell(ci, old, new, required=True):
    global n_edits
    src = "".join(nb["cells"][ci]["source"])
    cnt = src.count(old)
    if cnt != 1:
        if required:
            raise SystemExit(f"[FAIL] cell {ci}: expected 1 of {old!r}, found {cnt}")
        return
    nb["cells"][ci]["source"] = (src.replace(old, new)).splitlines(keepends=True)
    n_edits += 1


# --- markdown cell 0: fresh v56 description ---
nb["cells"][0]["source"] = (
    "# Multimodal Cancer Challenge 2026 — v56: L1+L2 (ensemble teacher + heavier noise)\n"
    "\n"
    "**Goal: a new high *single* model** on the proven architecture (dual EfficientNet-B0,\n"
    "late fusion — the exact config that scored LB 0.8355). Two levers vs v47:\n"
    "\n"
    "- **L1 — distil an *ensemble* teacher**, not the lucky seed-2. The teacher is the\n"
    "  probability-mean of the three v47 seeds (smoother, lower-variance soft targets →\n"
    "  better generalization). Falls back to `submissionv47seed2` automatically if the\n"
    "  ensemble dataset isn't attached.\n"
    "- **L2 — heavier student noise** (the regularizer that matters on a 12-patient task):\n"
    "  RandomErasing 0.25→0.35, ColorJitter 0.4→0.5, Dropout 0.4→0.45. Same tested code\n"
    "  paths, stronger settings.\n"
    "\n"
    "**Teacher dataset:** upload `results/teacher_ensemble/teacher_v47seeds_mean.csv` as a\n"
    "Kaggle dataset named **`teacherv47ensemble`** (so the path resolves). Expect the load\n"
    "log to print **mean soft target ≈ 0.49** (the ensemble), not 0.42 (seed-2).\n"
    "\n"
    "*Second-wave / budget-permitting run — start the cross-arch fleet (v52/v48/v49) and the\n"
    "CPU L4 GBM first.*\n"
).splitlines(keepends=True)
n_edits += 1

# --- config edits (cell 2) ---
edit_cell(2, "SEEDS               = [401, 402, 403, 404]                  # v50: LOTTERY backbone seeds",
          "SEEDS               = [1, 2]                                  # v56: L1+L2 (ensemble teacher + heavy noise)")
edit_cell(2, "RANDOM_ERASING_P    = 0.25", "RANDOM_ERASING_P    = 0.35")
edit_cell(2, "DROPOUT             = 0.4", "DROPOUT             = 0.45")

# L1: prepend the ensemble-teacher candidates (seed2 stays as fallback)
edit_cell(2,
    '_PSEUDO_LABEL_CANDIDATES = [\n    "/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv",',
    '_PSEUDO_LABEL_CANDIDATES = [\n'
    '    "/kaggle/input/datasets/rafaelproena/teacherv47ensemble/teacher_v47seeds_mean.csv",\n'
    '    "/kaggle/input/teacherv47ensemble/teacher_v47seeds_mean.csv",\n'
    '    "/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv",')

# --- L2: stronger photometric jitter (cell with train_modality_transform) ---
aug_ci = next(ci for ci, c in enumerate(nb["cells"])
              if c["cell_type"] == "code" and "def train_modality_transform" in "".join(c["source"]))
edit_cell(aug_ci, "T.ColorJitter(brightness=0.4, contrast=0.4)",
          "T.ColorJitter(brightness=0.5, contrast=0.5)")

# --- teacher metadata label (cosmetic, keep honest) ---
meta_ci = next((ci for ci, c in enumerate(nb["cells"])
                if c["cell_type"] == "code" and '"teacher":' in "".join(c["source"])), None)
if meta_ci is not None:
    edit_cell(meta_ci, '"teacher": "v46_ensemble_lb0.8236"', '"teacher": "v47_3seed_ensemble_mean"', required=False)

# --- relabel v50 -> v56 in remaining code prints/comments ---
for ci, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    src2 = src.replace("v50", "v56").replace("LOTTERY", "L1+L2").replace(
        "safe lottery", "L1+L2 backup").replace("lottery", "L1+L2 backup")
    if src2 != src:
        nb["cells"][ci]["source"] = src2.splitlines(keepends=True)

# normalize every cell's source to list-of-lines form (nbformat allows str OR list;
# v50's untouched cells are strings -- make them all uniform lists)
for c in nb["cells"]:
    c["source"] = "".join(c["source"]).splitlines(keepends=True)

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"applied {n_edits} required edits -> {OUT.name}")

# --- validate: structure + each code cell compiles + key edits present ---
re_nb = json.loads(OUT.read_text(encoding="utf-8"))
for i, c in enumerate(re_nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])  # robust whether source is str or list
        pysrc = "\n".join(ln for ln in src.split("\n") if not ln.lstrip().startswith(("!", "%")))
        compile(pysrc, f"cell{i}", "exec")
joined = "\n".join("".join(c["source"]) for c in re_nb["cells"])
assert "SEEDS               = [1, 2]" in joined
assert "RANDOM_ERASING_P    = 0.35" in joined
assert "DROPOUT             = 0.45" in joined
assert "teacher_v47seeds_mean.csv" in joined
assert "brightness=0.5, contrast=0.5" in joined
assert 'timm.create_model("efficientnet_b0"' in joined  # still B0 late fusion
print("VALID: all code cells compile; L1+L2 edits confirmed; backbone still efficientnet_b0")
