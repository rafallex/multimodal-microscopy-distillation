"""v51 = ConvNeXt-Tiny + soft pseudo from v47_s2 — the 2nd diverse ensemble member.

Why ConvNeXt-Tiny (over RegNetY) as backup diversity:
  * Strong: 82.1% ImageNet top-1 (vs ResNet-50's 76%) -> a much stronger solo model
    than ResNet-50, so less risk of dragging the cross-arch blend.
  * Maximally diverse from our fleet: ConvNeXt has NO squeeze-excite, uses LayerNorm
    + GELU + large-kernel (7x7) depthwise convs and a patchify stem. EfficientNet
    (MBConv + SE) and RegNetY (SE) are far more similar to each other. ConvNeXt is
    the most architecturally distinct strong backbone available -> lowest expected
    correlation with v47_s2 = biggest ensemble headroom.

This needs a GENERIC 1-channel stem adapter (ConvNeXt's first conv is stem.0, a
4x4 stride-4 patchify conv, not EffNet's conv_stem). The adapter finds the first
Conv2d in the timm model and averages its pretrained RGB weights to 1 channel —
works for any architecture. Smoke-tested locally with torch before any GPU run.

Same v47_s2 teacher and recipe as v48/v49/v50; only the backbone differs.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
V47 = ROOT / "notebooks" / "improvedv47_source.ipynb"
OUT = ROOT / "notebooks" / "improvedv51_source.ipynb"

nb = json.loads(V47.read_text(encoding="utf-8"))

# ── cell 0: markdown header ──
nb["cells"][0]["source"] = ("""# Multimodal Cancer Classification Challenge 2026 — v51: ConvNeXt-Tiny + soft pseudo (v47_s2 teacher)

**v51 = the v47 soft-pseudo recipe with a ConvNeXt-Tiny backbone — the 2nd diverse
member for the cross-architecture ensemble** (backup for ResNet-50, which may be too
weak alone).

## Why ConvNeXt-Tiny

Our entire fleet is EfficientNet-B0 (rank-corr >0.94 with v47_s2), so blending them is
capped. ResNet-50 (v49) is decorrelated (0.795) but weak (~0.72 base). ConvNeXt-Tiny is
the best of both:

| | ImageNet top-1 | SE blocks? | stem | vs EffNet |
|---|---|---|---|---|
| EfficientNet-B0/B2 | 77 / 80% | yes | 3x3 s2 | — |
| ResNet-50 | 76% | no | 7x7 s2 | moderate diff |
| **ConvNeXt-Tiny** | **82%** | **no** | **4x4 s4 patchify** | **max diff (LayerNorm/GELU/7x7 dwconv)** |

So ConvNeXt-Tiny should be both **stronger than ResNet-50** and **more decorrelated
than anything** — the ideal third leg for a blend that clears 0.85.

## What v51 changes vs v47

| Component | v47 | **v51** |
|---|---|---|
| Backbone | EfficientNet-B0 | **ConvNeXt-Tiny** (`convnext_tiny`, timm) |
| 1-ch stem | conv_stem averaged | **generic: first Conv2d averaged to 1-ch** (smoke-tested) |
| Batch size | 128 | **64** (ConvNeXt-Tiny memory on T4) |
| Pseudo teacher | v46 ens (0.8236) | **v47_seed2 (0.8355)** |
| Seeds | [1,2,3] | **[1,2,3]** |
| Everything else | — | identical (soft pseudo all 59k, MIL, AdaBN, SWA, 40-way TTA) |

## Required Kaggle input

`rafaelproena/a3-adl` + **`rafaelproena/submissionv47seed2`** (same dataset as v48/v49/v50).
Internet ON so the ConvNeXt-Tiny ImageNet weights download.

## Sanity check before the ~9 h commit (cell 4 self-test runs in seconds)

```
Backbone: ConvNeXt-Tiny   |   feature dim: 768   |   params: ~57M
Output shape: torch.Size([2])
```
If it errors or prints the wrong backbone, abort before training. The teacher log must
show `submissionv47seed2/submission_seed2.csv` and mean soft target ~0.42.

## Outputs / final use

Per-seed `submission_seed{N}.csv` (incremental saves; checkpoints persist). Feed the
best seed into `build_cross_arch_ensemble.py` as the ConvNeXt member. The target blend:
v47_s2 (EffNet-B0) + v48 (EffNet-B2) + v49 (ResNet-50) + **v51 (ConvNeXt-Tiny)** — four
families, the most diverse ensemble we can build.

## Compute

~9 h on T4 x2 (batch 64, 3 seeds). Run AFTER v48 (it's the higher-capacity bet); use
v51 if v49's ResNet-50 comes in too weak (<0.80) to help the blend.
""").splitlines(keepends=True)

# ── cell 2: config ──
src = "".join(nb["cells"][2]["source"])
for old, new in [
    ('"/kaggle/input/datasets/rafaelproena/submissionv46/submission.csv"',
     '"/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv"'),
    ('"/kaggle/input/submissionv46/submission.csv"',
     '"/kaggle/input/submissionv47seed2/submission_seed2.csv"'),
    ('"/kaggle/input/v46-predictions/submission.csv"',
     '"/kaggle/input/v47seed2-predictions/submission_seed2.csv"'),
    ("SEEDS               = [1, 2, 3]                  # multi-seed ensemble",
     "SEEDS               = [1, 2, 3]                  # v51: ConvNeXt-Tiny seeds"),
    ("BATCH_SIZE  = 128",
     "BATCH_SIZE  = 64     # v51: sized for ConvNeXt-Tiny memory on T4"),
    ("USE_EFFICIENTNET    = True",
     'USE_EFFICIENTNET    = True\nBACKBONE_NAME       = "convnext_tiny"   # v51: generic timm backbone (overrides _make_branch)'),
]:
    assert old in src, f"cell2 sub not found: {old[:50]}"
    src = src.replace(old, new)
nb["cells"][2]["source"] = src.splitlines(keepends=True)

# ── cell 5: fix the stale fallback-warning dataset name ──
src5 = "".join(nb["cells"][5]["source"])
src5 = src5.replace(
    "print(f\"  Attach the 'submissionv46' Kaggle dataset to enable pseudo-labels.\")",
    "print(f\"  Attach the 'submissionv47seed2' Kaggle dataset to enable pseudo-labels.\")")
nb["cells"][5]["source"] = src5.splitlines(keepends=True)

# ── cell 4: inject generic timm branch + 1-ch adapter, override _make_branch ──
src4 = "".join(nb["cells"][4]["source"])
helpers = '''def _set_submodule(root, dotted, new):
    """Replace a possibly-nested submodule (e.g. 'stem.0') with `new`."""
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new
    else:
        setattr(obj, last, new)

def _make_timm_branch(model_name, pretrained=True):
    """Generic single-modality branch: any timm model, first Conv2d adapted to 1-ch."""
    import timm
    net = timm.create_model(model_name, pretrained=pretrained,
                            num_classes=0, global_pool="avg")
    first_name, first = next((n, m) for n, m in net.named_modules()
                             if isinstance(m, nn.Conv2d))
    new_conv = nn.Conv2d(1, first.out_channels, kernel_size=first.kernel_size,
                         stride=first.stride, padding=first.padding,
                         dilation=first.dilation, groups=1,
                         bias=first.bias is not None)
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(first.weight.mean(dim=1, keepdim=True))
            if first.bias is not None:
                new_conv.bias.copy_(first.bias)
    _set_submodule(net, first_name, new_conv)
    return net, net.num_features

'''
# insert helpers right before _make_branch
anchor = "def _make_branch(pretrained=True):"
assert anchor in src4
src4 = src4.replace(anchor, helpers + anchor)
# override _make_branch body to use the generic timm backbone
src4 = src4.replace(
    "def _make_branch(pretrained=True):\n    if USE_EFFICIENTNET:\n        return _make_effnet_b0_branch(pretrained)\n    return _make_resnet18_branch(pretrained)",
    'def _make_branch(pretrained=True):\n    # v51: generic timm backbone (ConvNeXt-Tiny) via BACKBONE_NAME\n    return _make_timm_branch(BACKBONE_NAME, pretrained)')
# self-test print -> show backbone name + feature dim
src4 = src4.replace(
    'print(f"Backbone: {\'EfficientNet-B0\' if USE_EFFICIENTNET else \'ResNet-18\'}")',
    'print(f"Backbone: {BACKBONE_NAME}   |   feature dim: {_m.bf_branch[1] if isinstance(_m.bf_branch, tuple) else _m.head[0].in_features // 2}")')
nb["cells"][4]["source"] = src4.splitlines(keepends=True)

# ── cell 7: fix misleading log ──
src7 = "".join(nb["cells"][7]["source"])
src7 = src7.replace(
    'print(f"  soft pseudo loss weight = {PSEUDO_LOSS_WEIGHT}  (Hinton 2015 distillation, teacher = v46)")',
    'print(f"  soft pseudo loss weight = {PSEUDO_LOSS_WEIGHT}  (teacher = v47_seed2, backbone ConvNeXt-Tiny)")')
nb["cells"][7]["source"] = src7.splitlines(keepends=True)

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# verify
re_nb = json.loads(OUT.read_text(encoding="utf-8"))
c2 = "".join(re_nb["cells"][2]["source"]); c4 = "".join(re_nb["cells"][4]["source"])
assert 'BACKBONE_NAME       = "convnext_tiny"' in c2
assert "submissionv47seed2/submission_seed2.csv" in c2 and "BATCH_SIZE  = 64" in c2
assert "_make_timm_branch" in c4 and "_set_submodule" in c4
assert "return _make_timm_branch(BACKBONE_NAME, pretrained)" in c4
print(f"Saved {OUT.name}  ({len(re_nb['cells'])} cells)")
print("ConvNeXt-Tiny, batch 64, teacher v47_s2, generic 1-ch stem adapter injected + verified.")
