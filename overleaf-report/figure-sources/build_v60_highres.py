"""Builder for v60 = EfficientNet-B0 @ 192px (the resolution lever).

EfficientNet-only, single-model, no ensembles. Same proven recipe + backbone (dual
EffNet-B0 late fusion, soft-pseudo distillation, SWA, AdaBN, MIL aux) -- the ONE change
is input resolution: train upscaled 128 -> 192 (EffNet-B0's pretraining is nearer 224,
and 128px loses cell detail), with the 40-way TTA shifted to scales around 192. Base:
v50. L1 ensemble teacher (seed-2 fallback). Batch dropped to 48 for the 2.25x pixels.
CPU-smoke-tested (transform 128->192, model forward+learn at 192).

Emits notebooks/improvedv60_source.ipynb. Untested on GPU by design.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "notebooks" / "improvedv50_sourceDO_NOT_RUN_.ipynb"
OUT = ROOT / "notebooks" / "improvedv60_source.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))
n = 0
skipped = []


def edit(loc, old, new, required=True):
    global n
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if loc in s and old in s:
            c["source"] = s.replace(old, new).splitlines(keepends=True)
            n += 1
            return True
    if required:
        raise SystemExit(f"[FAIL] required edit not found: {old!r}")
    skipped.append(old)
    return False


# --- resolution: TTA scales around 192 + add TRAIN_RES config ---
edit("TTA_SCALES",
     "TTA_SCALES          = (96, 112, 128, 144, 160)   # 5 scales x 8 D4 = 40-way",
     "TTA_SCALES          = (144, 168, 192, 216, 240)  # v60: 5 scales around 192px x 8 D4 = 40-way\n"
     "TRAIN_RES           = 192                         # v60: upscale 128 -> 192 (EffNet native scale)")
# --- train at 192: prepend a Resize to the modality transform pipeline ---
edit("return T.Compose(steps)", "return T.Compose(steps)",
     "return T.Compose([T.Resize((TRAIN_RES, TRAIN_RES))] + steps)")
# --- batch down for 2.25x pixels ---
edit("BATCH_SIZE", "BATCH_SIZE  = 128    # v50: sized for EfficientNet-B0 memory on T4",
     "BATCH_SIZE  = 48     # v60: EfficientNet-B0 @ 192px on T4")
# --- seeds ---
edit("SEEDS", "SEEDS               = [401, 402, 403, 404]", "SEEDS               = [1, 2]")
# --- self-test at the training resolution ---
edit("_x = torch.zeros", "_x = torch.zeros(2, 1, 128, 128)", "_x = torch.zeros(2, 1, 192, 192)", required=False)
# --- honest version labels (backbone stays EfficientNet-B0, which is correct) ---
edit("=== v50: EfficientNet-B0",
     "# === v50: EfficientNet-B0 + soft pseudo from v47_seed2 (LB 0.8355) ===",
     "# === v60: EfficientNet-B0 @ 192px + soft pseudo from v47_seed2 (resolution lever) ===")
edit("Config (v50", "Config (v50 - EfficientNet-B0 + soft pseudo from v47_seed2):",
     "Config (v60 - EfficientNet-B0 @ 192px + soft pseudo from v47_seed2):")
edit("Training seed=", "=== v47: Training seed=", "=== v60: Training seed=")
# --- L1 ensemble teacher (seed-2 fallback) ---
edit('_PSEUDO_LABEL_CANDIDATES',
     '_PSEUDO_LABEL_CANDIDATES = [\n    "/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv",',
     '_PSEUDO_LABEL_CANDIDATES = [\n'
     '    "/kaggle/input/datasets/rafaelproena/teacherv47ensemble/teacher_v47seeds_mean.csv",\n'
     '    "/kaggle/input/teacherv47ensemble/teacher_v47seeds_mean.csv",\n'
     '    "/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv",')
edit('"teacher":', '"teacher": "v46_ensemble_lb0.8236"', '"teacher": "v47_3seed_ensemble_mean"', required=False)

nb["cells"][0]["source"] = (
    "# Multimodal Cancer Challenge 2026 - v60: EfficientNet-B0 @ 192px (resolution lever)\n"
    "\n"
    "**EfficientNet-only, single model, no ensembles.** Identical proven recipe and backbone\n"
    "as v47 (dual EfficientNet-B0 late fusion + soft-pseudo distillation + SWA + AdaBN + MIL\n"
    "aux) - the **one change is input resolution**: train images upscaled **128 -> 192px**\n"
    "(EffNet-B0 is pretrained nearer 224, and 128px crops lose cell detail), with the 40-way\n"
    "TTA shifted to scales around 192 `(144,168,192,216,240)`. The last genuinely-untapped\n"
    "single-model knob.\n"
    "\n"
    "Batch 48 (2.25x the pixels), seeds [1,2], L1 ensemble teacher (seed-2 auto-fallback).\n"
    "Submit the per-seed CSVs (`submission_seed1/2.csv`), keep the best single seed.\n"
    "CPU-smoke-tested (transform 128->192, model forward+learn at 192).\n"
).splitlines(keepends=True)

for c in nb["cells"]:
    c["source"] = "".join(c["source"]).splitlines(keepends=True)

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"applied {n} edits ({len(skipped)} optional skipped) -> {OUT.name}")
for s in skipped:
    print("  SKIPPED:", s[:70])

re_nb = json.loads(OUT.read_text(encoding="utf-8"))
for i, c in enumerate(re_nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        pysrc = "\n".join(ln for ln in src.split("\n") if not ln.lstrip().startswith(("!", "%")))
        compile(pysrc, f"cell{i}", "exec")
code = "\n".join("".join(c["source"]) for c in re_nb["cells"] if c["cell_type"] == "code")
assert "TRAIN_RES           = 192" in code
assert "T.Resize((TRAIN_RES, TRAIN_RES))" in code
assert "(144, 168, 192, 216, 240)" in code
assert "BATCH_SIZE  = 48" in code
assert "SEEDS               = [1, 2]" in code
assert 'timm.create_model("efficientnet_b0"' in code  # backbone unchanged (correct)
assert "teacher_v47seeds_mean.csv" in code
assert "=== v47: Training" not in code and "Config (v50" not in code  # no stale version banner
print("VALID: train@192 (Resize+TRAIN_RES), TTA~192, batch 48, B0 backbone, L1 teacher, clean labels")
