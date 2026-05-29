"""Generate v48/v49/v50 from proven v47, pulling the two levers we never pulled
with pseudo-labels: BACKBONE CAPACITY (v48) and ARCHITECTURE DIVERSITY (v49).

Why (from results/ analysis on 2026-05-28):
  * Every strong model we have is EfficientNet-B0 -> rank-corr >0.94 with v47_s2,
    so within-family ensembling is capped (all 8 CPU probes lost to pure s2).
  * The ONLY decorrelated model we ever trained is v34 ResNet-50 (corr 0.795),
    but it never saw pseudo-labels.
  * Training AUC is anti-predictive of LB (v47_s2 had the LOWEST tr_auc, HIGHEST
    LB) -> seed choice must go through the public LB.
  * Leaders sit at 0.848 with FEWER entries than us -> they have a stronger core,
    most plausibly more backbone capacity. We are on the smallest EfficientNet.

Plan (all distill from the v47_s2 teacher = best teacher we have, LB 0.8355):
  v48 = EfficientNet-B2  (CAPACITY lever; batch 96 for memory)   seeds [1,2,3]
  v49 = ResNet-50        (DIVERSITY lever; batch 64 for memory)  seeds [1,2,3]
  v50 = EfficientNet-B0  (safe seed LOTTERY)                     seeds [401-404]

Then CPU-ensemble best single seeds ACROSS architectures (build_cross_arch_ensemble.py)
- the diversity makes a weighted average able to exceed any single member, the one
path to >0.85 from our assets.

Backbone swap is safe: the fusion head sizes itself from net.num_features, and the
PatientBalancedSampler asserts batch_size % patients_per_batch == 0 (96/4, 64/4 ok),
OneCycleLR uses len(train_loader). Only the model-name string + batch size change.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
V47 = ROOT / "notebooks" / "improvedv47_source.ipynb"

TEACHER_SUBS = [
    ('"/kaggle/input/datasets/rafaelproena/submissionv46/submission.csv"',
     '"/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv"'),
    ('"/kaggle/input/submissionv46/submission.csv"',
     '"/kaggle/input/submissionv47seed2/submission_seed2.csv"'),
    ('"/kaggle/input/v46-predictions/submission.csv"',
     '"/kaggle/input/v47seed2-predictions/submission_seed2.csv"'),
]

HEADER = """# Multimodal Cancer Classification Challenge 2026 — {ver}: {backbone_name} + soft pseudo (v47_s2 teacher)

**{ver} = the v47 soft-pseudo recipe, but with the {lever} lever: {backbone_name} backbone.**

## Why (read the results-folder analysis first)

Rank-correlation of every model we have with our best (v47_s2, LB 0.8355):

| model | backbone | rank-corr w/ v47_s2 |
|---|---|---|
| v47_ens / v46 / v44 | EfficientNet-B0 | 0.94 – 0.98 |
| v41 | EfficientNet-B0 | 0.89 |
| **v34** | **ResNet-50** | **0.795 (only decorrelated model)** |

Every strong model is an EfficientNet-B0 clone, so averaging them is capped — that
is exactly why all 8 CPU ensemble probes lost to the pure v47_s2 single seed.
**To average ABOVE a single model you need diversity AND strength.** Two levers we
never pulled while using pseudo-labels:

- **Capacity** (v48 = EfficientNet-B2): v34 concluded "capacity isn't the bottleneck",
  but that test was supervised-only on 114k cells. With the soft-pseudo set the model
  now trains on 173k cells — a bigger backbone is finally justified.
- **Diversity** (v49 = ResNet-50): the one architecture decorrelated from our fleet.
  Trained with pseudo-labels it becomes a strong-AND-different member that can lift a
  cross-architecture ensemble past any single EffNet.

{lever_note}

## What {ver} changes vs v47

| Component | v47 | **{ver}** |
|---|---|---|
| Backbone | EfficientNet-B0 (per branch) | **{backbone_name}** |
| Batch size | 128 | **{batch}**{batch_note} |
| Pseudo teacher | v46 ensemble (0.8236) | **v47_seed2 (0.8355)** |
| Seeds | [1,2,3] | **{seeds}** |
| Everything else | — | identical (soft pseudo all 59k cells, MIL, AdaBN, SWA, 40-way TTA) |

## Required Kaggle input (same dataset as the other harvest notebooks)

`rafaelproena/a3-adl` + **`rafaelproena/submissionv47seed2`** (upload
`results/v47/submission_seed2.csv` once, private). Internet must be ON so the
{backbone_name} ImageNet weights download.

## Sanity check before the ~{hours} h commit

```
Backbone: {backbone_name}             <- the model-build cell must print this
Pseudo-labels loaded from .../submissionv47seed2/submission_seed2.csv  [mode: SOFT]
  mean soft target: 0.42??   (train was 0.3876)
  Combined training set: 173342 cells (114302 real + 59040 pseudo)
```
If it prints EfficientNet-B0, or the teacher path contains `submissionv46`, **abort**.

## Outputs and what matters

Per-seed `submission_seed{{N}}.csv` (saved incrementally — a timeout only loses the
in-progress seed; checkpoints `runs/swa_seed{{N}}.pt` also persist). The single most
useful artifact is **each per-seed CSV**, for the cross-architecture ensemble below.

## After v48 + v49 + v50 land: the cross-architecture ensemble (CPU, no GPU)

Run `build_cross_arch_ensemble.py`. It grid-searches weighted averages of the best
single seed from each architecture:
   w1 * v47_s2(EffNet-B0) + w2 * v48_best(EffNet-B2) + w3 * v49_best(ResNet-50)
Because these are genuinely decorrelated (~0.80 across families), the weighted blend
can exceed every individual member — the realistic path to >0.85 and #1.

Submit BOTH the best single seed AND the best cross-arch blend as your 2 final picks.

## Compute

~{hours} h on Kaggle T4 x2. Run order for max shot at 0.85: **v48 (capacity) ->
v49 (diversity) -> v50 (lottery)**. v48 and v49 are the swings; v50 is the safe net.
"""


def build(ver, out_name, *, backbone, lever, seeds, batch, use_effnet,
          backbone_name, hours, lever_note):
    nb = json.loads(V47.read_text(encoding="utf-8"))
    n = 0

    # ---- cell 0: markdown ----
    batch_note = f"  — reduced from 128 for {backbone_name} memory" if batch != 128 else ""
    nb["cells"][0]["source"] = HEADER.format(
        ver=ver, backbone_name=backbone_name, lever=lever, lever_note=lever_note,
        batch=batch, batch_note=batch_note, seeds=str(seeds), hours=hours,
    ).splitlines(keepends=True)

    # ---- cell 2: config ----
    src = "".join(nb["cells"][2]["source"])
    for old, new in TEACHER_SUBS:
        if old in src:
            src = src.replace(old, new); n += 1

    src = src.replace(
        "SEEDS               = [1, 2, 3]                  # multi-seed ensemble",
        f"SEEDS               = {seeds}                  # {ver}: {lever} backbone seeds"); n += 1

    src = src.replace(
        "BATCH_SIZE  = 128",
        f"BATCH_SIZE  = {batch}    # {ver}: sized for {backbone_name} memory on T4"); n += 1

    if not use_effnet:
        src = src.replace(
            "USE_EFFICIENTNET    = True",
            "USE_EFFICIENTNET    = False   # v49: ResNet-50 branch (diversity lever)"); n += 1

    src = src.replace(
        "# === v47: noisy student ROUND 2 — soft pseudo-labels from v46 (LB 0.8236) ===\n"
        "# v46 reached LB 0.8236 (current #1). Its predictions are a much stronger teacher\n"
        "# than v44_seed1 was (LB 0.7844). Per Xie 2020, iterative noisy student gains\n"
        "# accuracy across rounds. This is round 2.",
        f"# === {ver}: {backbone_name} + soft pseudo from v47_seed2 (LB 0.8355) ===\n"
        f"# {lever} lever. Teacher = v47_seed2 (best single model). Backbone swapped to\n"
        f"# {backbone_name}; head auto-sizes from num_features. Distilling a different\n"
        f"# architecture from v47_s2's soft targets for a decorrelated, strong ensemble member."); n += 1

    src = src.replace(
        'print(f"\\nConfig (v47 - noisy student ROUND 2: soft pseudo from v46 / Hinton distillation):")',
        f'print(f"\\nConfig ({ver} - {backbone_name} + soft pseudo from v47_seed2):")'); n += 1

    src = src.replace(
        "print(f\"  Attach the 'submissionv46' Kaggle dataset to enable pseudo-labels.\")",
        "print(f\"  Attach the 'submissionv47seed2' Kaggle dataset to enable pseudo-labels.\")")

    nb["cells"][2]["source"] = src.splitlines(keepends=True)

    # ---- cell 4: backbone string ----
    src4 = "".join(nb["cells"][4]["source"])
    if use_effnet and backbone != "efficientnet_b0":
        src4 = src4.replace('timm.create_model("efficientnet_b0"',
                            f'timm.create_model("{backbone}"'); n += 1
        # keep the printed backbone label honest (else it still says EfficientNet-B0)
        src4 = src4.replace("'EfficientNet-B0' if USE_EFFICIENTNET else 'ResNet-18'",
                            f"'{backbone_name}' if USE_EFFICIENTNET else 'ResNet-18'"); n += 1
    if not use_effnet:
        # _make_resnet18_branch -> resnet50 (conv1 + fc.in_features handled dynamically)
        src4 = src4.replace("net = models.resnet18(weights=weights)",
                            "net = models.resnet50(weights=weights)"); n += 1
        src4 = src4.replace("'EfficientNet-B0' if USE_EFFICIENTNET else 'ResNet-18'",
                            "'EfficientNet-B0' if USE_EFFICIENTNET else 'ResNet-50'")
    nb["cells"][4]["source"] = src4.splitlines(keepends=True)

    # ---- cell 7: fix misleading log string + saved backbone metadata ----
    src7 = "".join(nb["cells"][7]["source"])
    src7 = src7.replace(
        'print(f"  soft pseudo loss weight = {PSEUDO_LOSS_WEIGHT}  (Hinton 2015 distillation, teacher = v46)")',
        f'print(f"  soft pseudo loss weight = {{PSEUDO_LOSS_WEIGHT}}  (teacher = v47_seed2, backbone {backbone_name})")')
    # the saved checkpoint metadata also hard-codes the template backbone — correct it
    if use_effnet and backbone != "efficientnet_b0":
        src7 = src7.replace('"backbone": "efficientnet_b0"', f'"backbone": "{backbone}"'); n += 1
    if not use_effnet:
        src7 = src7.replace('else "resnet18"', 'else "resnet50"'); n += 1
    nb["cells"][7]["source"] = src7.splitlines(keepends=True)

    out = ROOT / "notebooks" / out_name
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- verify ----
    re_nb = json.loads(out.read_text(encoding="utf-8"))
    c2 = "".join(re_nb["cells"][2]["source"]); c4 = "".join(re_nb["cells"][4]["source"])
    assert "submissionv47seed2/submission_seed2.csv" in c2 and "submissionv46" not in c2
    assert f"SEEDS               = {seeds}" in c2
    assert f"BATCH_SIZE  = {batch}" in c2
    if use_effnet:
        assert f'timm.create_model("{backbone}"' in c4, f"{ver}: effnet name not set"
        assert "USE_EFFICIENTNET    = True" in c2
    else:
        assert "models.resnet50(weights=weights)" in c4, f"{ver}: resnet50 not set"
        assert "USE_EFFICIENTNET    = False" in c2
    print(f"  {out_name:<26} {backbone_name:<16} batch={batch} seeds={seeds}  ({n} subs, verified)")


print("Building backbone-diversity notebooks from proven v47:")
build("v48", "improvedv48_source.ipynb", backbone="efficientnet_b2", lever="CAPACITY",
      seeds=[1, 2, 3], batch=96, use_effnet=True, backbone_name="EfficientNet-B2", hours=9,
      lever_note="**{ver} pulls the capacity lever.** EfficientNet-B2 (~9.2M params/branch vs "
                 "B0's 5.3M) on the pseudo-expanded 173k-cell set.".replace("{ver}", "v48"))
build("v49", "improvedv49_source.ipynb", backbone="resnet50", lever="DIVERSITY",
      seeds=[1, 2, 3], batch=64, use_effnet=False, backbone_name="ResNet-50", hours=9,
      lever_note="**v49 pulls the diversity lever.** ResNet-50 is the only architecture "
                 "decorrelated (0.795) from our EffNet fleet — the key ensemble ingredient.")
build("v50", "improvedv50_source.ipynb", backbone="efficientnet_b0", lever="LOTTERY",
      seeds=[401, 402, 403, 404], batch=128, use_effnet=True, backbone_name="EfficientNet-B0", hours=8,
      lever_note="**v50 is the safe lottery.** Proven B0 recipe, fresh seeds vs the v47_s2 "
                 "teacher — cheap insurance and more ensemble members if v48/v49 underdeliver.")
print("\nAll three verified. Teacher=v47_seed2; v48=B2(cap), v49=RN50(div), v50=B0(lottery).")
print("Run order: v48 -> v49 -> v50. Then build_cross_arch_ensemble.py for the >0.85 blend.")
