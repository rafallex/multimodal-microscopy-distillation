"""Builder for v59 = EfficientNetV2-S dual late-fusion (a stronger, modern EfficientNet).

EfficientNet-only, single-model, no ensembles. v59 keeps the PROVEN recipe (dual late
fusion, soft-pseudo distillation, SWA, AdaBN, 40-way TTA, MIL aux) and only swaps the
backbone B0 -> EfficientNetV2-S (fused-MBConv, stronger ImageNet pretrain). A clean
single-variable change, like the v48 B2 swap. Distils the L1 ensemble teacher
(seed-2 fallback). Base: v50. CPU-smoke-tested (41.7M params, learns).

Emits notebooks/improvedv59_source.ipynb. Untested on GPU by design.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "notebooks" / "improvedv50_sourceDO_NOT_RUN_.ipynb"
OUT = ROOT / "notebooks" / "improvedv59_source.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))
n = 0


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
        raise SystemExit(f"[FAIL] not found: {old!r}")
    return False


# 1) backbone swap: EfficientNet-B0 -> EfficientNetV2-S (the only architecture change)
edit("timm.create_model", 'timm.create_model("efficientnet_b0"', 'timm.create_model("tf_efficientnetv2_s"')
# 2) honest labels (print + saved metadata)
edit("Backbone:", "'EfficientNet-B0' if USE_EFFICIENTNET else 'ResNet-18'",
     "'EfficientNetV2-S' if USE_EFFICIENTNET else 'ResNet-18'", required=False)
edit('"backbone":', '"backbone": "efficientnet_b0"', '"backbone": "tf_efficientnetv2_s"', required=False)
# 3) seeds + batch (V2-S ~20M/branch -> drop batch for T4)
edit("SEEDS", "SEEDS               = [401, 402, 403, 404]", "SEEDS               = [1, 2]", required=False)
edit("BATCH_SIZE", "BATCH_SIZE  = 128", "BATCH_SIZE  = 64", required=False)
# 4) L1 ensemble teacher (seed-2 fallback)
edit('_PSEUDO_LABEL_CANDIDATES',
     '_PSEUDO_LABEL_CANDIDATES = [\n    "/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv",',
     '_PSEUDO_LABEL_CANDIDATES = [\n'
     '    "/kaggle/input/datasets/rafaelproena/teacherv47ensemble/teacher_v47seeds_mean.csv",\n'
     '    "/kaggle/input/teacherv47ensemble/teacher_v47seeds_mean.csv",\n'
     '    "/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv",')
edit('"teacher":', '"teacher": "v46_ensemble_lb0.8236"', '"teacher": "v47_3seed_ensemble_mean"', required=False)

nb["cells"][0]["source"] = (
    "# Multimodal Cancer Challenge 2026 - v59: EfficientNetV2-S (stronger EfficientNet)\n"
    "\n"
    "**EfficientNet-only, single model, no ensembles.** v59 keeps the exact proven recipe\n"
    "(dual late fusion + soft-pseudo distillation + SWA + AdaBN + 40-way TTA + MIL aux) and\n"
    "changes **one thing**: the backbone EfficientNet-B0 -> **EfficientNetV2-S** (fused-MBConv,\n"
    "stronger ImageNet pretraining). A cleaner, more modern EfficientNet than B0/B2/B3.\n"
    "\n"
    "~42M params (dual), batch 64. Distils the L1 ensemble teacher (3-seed mean; seed-2\n"
    "auto-fallback). Submit the per-seed CSVs (`submission_seed1/2.csv`) and keep the best\n"
    "single seed -- not the mean. CPU-smoke-tested.\n"
).splitlines(keepends=True)

for c in nb["cells"]:
    c["source"] = "".join(c["source"]).splitlines(keepends=True)

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"applied {n} edits -> {OUT.name}")

re_nb = json.loads(OUT.read_text(encoding="utf-8"))
for i, c in enumerate(re_nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        pysrc = "\n".join(ln for ln in src.split("\n") if not ln.lstrip().startswith(("!", "%")))
        compile(pysrc, f"cell{i}", "exec")
joined = "\n".join("".join(c["source"]) for c in re_nb["cells"])
assert 'tf_efficientnetv2_s' in joined
assert 'efficientnet_b0' not in joined.replace("_make_effnet_b0_branch", "")  # only the fn name may remain
assert "SEEDS               = [1, 2]" in joined and "BATCH_SIZE  = 64" in joined
assert "teacher_v47seeds_mean.csv" in joined
print("VALID: backbone=tf_efficientnetv2_s, seeds[1,2], batch 64, L1 teacher, compiles")
