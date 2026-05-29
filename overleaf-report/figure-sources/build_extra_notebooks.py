"""Local generator -> 3 more KAGGLE notebooks to widen the parallel batch.

  v53 = ConvNeXt-Tiny + EARLY fusion  (double-diverse: new backbone AND new fusion)
  v54 = RegNetY-3.2GF + late fusion   (another strong family)
  v55 = EfficientNet-B3 + late fusion (more capacity than B2)

All derive from the proven v47 notebook (pseudo machinery / data / training / TTA
unchanged); only the model cell (4) + a few config constants change. All distill
from the v47_s2 teacher. Each is smoke-tested locally by smoketest_backbones.py.
"""
import json, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
NB = ROOT / "notebooks"
V47 = NB / "improvedv47_source.ipynb"
V51 = NB / "improvedv51_source.ipynb"   # source of the generic timm dual-branch adapter

TEACHER = [
    ('"/kaggle/input/datasets/rafaelproena/submissionv46/submission.csv"',
     '"/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv"'),
    ('"/kaggle/input/submissionv46/submission.csv"',
     '"/kaggle/input/submissionv47seed2/submission_seed2.csv"'),
    ('"/kaggle/input/v46-predictions/submission.csv"',
     '"/kaggle/input/v47seed2-predictions/submission_seed2.csv"'),
]

# ── v53 model cell: GENERIC timm backbone, EARLY fusion (in_channels=2, single net) ──
V53_CELL4 = '''def _set_submodule(root, dotted, new):
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if last.isdigit(): obj[int(last)] = new
    else: setattr(obj, last, new)

def _make_timm_backbone(model_name, pretrained=True, in_channels=2):
    """Generic timm backbone with first Conv2d adapted to `in_channels` (early fusion uses 2)."""
    import timm
    net = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
    first_name, first = next((n, m) for n, m in net.named_modules() if isinstance(m, nn.Conv2d))
    new_conv = nn.Conv2d(in_channels, first.out_channels, kernel_size=first.kernel_size,
                         stride=first.stride, padding=first.padding, dilation=first.dilation,
                         groups=1, bias=first.bias is not None)
    if pretrained:
        with torch.no_grad():
            mean_rgb = first.weight.mean(dim=1, keepdim=True)            # [out,1,k,k]
            new_conv.weight.copy_(mean_rgb.repeat(1, in_channels, 1, 1) / in_channels)
            if first.bias is not None: new_conv.bias.copy_(first.bias)
    _set_submodule(net, first_name, new_conv)
    return net, net.num_features

# === v53: EARLY-FUSION single ConvNeXt-Tiny (cat[BF,FL] -> 2-ch -> one backbone) ===
class MultimodalClassifier(nn.Module):
    def __init__(self, pretrained=True, dropout=DROPOUT):
        super().__init__()
        self.backbone, fd = _make_timm_backbone(BACKBONE_NAME, pretrained, in_channels=2)
        hidden = 512 if fd >= 512 else 256
        self.head = nn.Sequential(
            nn.Linear(fd, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
    def forward(self, bf, fl):
        x = torch.cat([bf, fl], dim=1)
        return self.head(self.backbone(x)).squeeze(-1)

with torch.no_grad():
    _m = MultimodalClassifier(pretrained=False).cpu()
    _x = torch.zeros(2, 1, 128, 128)
    n_params = sum(p.numel() for p in _m.parameters())
    print(f"Output shape: {_m(_x, _x).shape}   params: {n_params / 1e6:.1f}M")
    print(f"Backbone: {BACKBONE_NAME} (EARLY-fusion, 2-ch input)")
    del _m, _x
'''


def base_v47():
    return json.loads(V47.read_text(encoding="utf-8"))


def patch_cell2(nb, *, seeds, batch, backbone_name=None, set_effnet=None, version, desc):
    src = "".join(nb["cells"][2]["source"])
    for o, n in TEACHER:
        src = src.replace(o, n)
    src = src.replace("SEEDS               = [1, 2, 3]                  # multi-seed ensemble",
                      f"SEEDS               = {seeds}                  # {version}: {desc} seeds")
    src = src.replace("BATCH_SIZE  = 128",
                      f"BATCH_SIZE  = {batch}    # {version}: sized for {desc} on T4")
    if backbone_name is not None:
        src = src.replace('USE_EFFICIENTNET    = True',
                          f'USE_EFFICIENTNET    = True\nBACKBONE_NAME       = "{backbone_name}"   # {version}: timm backbone')
    if set_effnet is False:
        src = src.replace("USE_EFFICIENTNET    = True", "USE_EFFICIENTNET    = False")
    src = src.replace('print(f"\\nConfig (v47 - noisy student ROUND 2: soft pseudo from v46 / Hinton distillation):")',
                      f'print(f"\\nConfig ({version} - {desc} + soft pseudo from v47_seed2):")')
    nb["cells"][2]["source"] = src.splitlines(keepends=True)
    # cell 5 + 7 cosmetic
    nb["cells"][5]["source"] = "".join(nb["cells"][5]["source"]).replace(
        "print(f\"  Attach the 'submissionv46' Kaggle dataset to enable pseudo-labels.\")",
        "print(f\"  Attach the 'submissionv47seed2' Kaggle dataset to enable pseudo-labels.\")").splitlines(keepends=True)
    nb["cells"][7]["source"] = "".join(nb["cells"][7]["source"]).replace(
        'print(f"  soft pseudo loss weight = {PSEUDO_LOSS_WEIGHT}  (Hinton 2015 distillation, teacher = v46)")',
        f'print(f"  soft pseudo loss weight = {{PSEUDO_LOSS_WEIGHT}}  (teacher = v47_seed2, {desc})")').splitlines(keepends=True)


def header(version, desc, extra):
    return (f"# Multimodal Cancer Challenge 2026 — {version}: {desc} + soft pseudo (v47_s2 teacher)\n\n"
            f"**{version} = the v47 soft-pseudo recipe with a {desc} model — extra diversity for "
            f"the cross-architecture ensemble.**\n\n{extra}\n\n"
            "## Required Kaggle input\n"
            "`rafaelproena/a3-adl` + **`rafaelproena/submissionv47seed2`** (same dataset as v48-v52).\n\n"
            "## Sanity check before commit (cell 4 self-test, seconds)\n"
            "The self-test must print the right backbone + a param count, and the teacher log must show "
            "`submissionv47seed2/...` with mean soft target ~0.42. Otherwise abort.\n\n"
            "## Output / use\nPer-seed `submission_seed{N}.csv` (incremental). Feed the best seed into "
            "`ensemble_crossarch_source.ipynb` as another member. Final picks = best blend + best single seed.\n").splitlines(keepends=True)


def finalize(nb, out_name):
    for c in nb["cells"]:
        c.setdefault("id", uuid.uuid4().hex[:8])
    (NB / out_name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    import nbformat
    nbformat.validate(nbformat.read(str(NB / out_name), as_version=4))
    return out_name


# ---- v53: ConvNeXt-Tiny EARLY fusion ----
nb = base_v47()
nb["cells"][0]["source"] = header("v53", "ConvNeXt-Tiny EARLY-fusion",
    "ConvNeXt-Tiny is our most diverse backbone; early fusion is our most diverse fusion. "
    "v53 combines BOTH for maximum decorrelation from the EffNet-B0 late-fusion fleet.")
nb["cells"][4]["source"] = V53_CELL4.splitlines(keepends=True)
patch_cell2(nb, seeds=[1, 2, 3], batch=96, backbone_name="convnext_tiny", version="v53",
            desc="ConvNeXt-Tiny EARLY-fusion")
finalize(nb, "improvedv53_source.ipynb")

# ---- v54: RegNetY-3.2GF late fusion (reuse v51 generic dual-branch adapter) ----
nb = base_v47()
nb["cells"][4]["source"] = json.loads(V51.read_text(encoding="utf-8"))["cells"][4]["source"]  # generic 1-ch dual-branch
nb["cells"][0]["source"] = header("v54", "RegNetY-3.2GF late-fusion",
    "RegNetY-3.2GF (81% ImageNet) is another strong family distinct from EffNet/ConvNeXt at the "
    "macro level — a cheap extra decorrelated member for a wide parallel batch.")
patch_cell2(nb, seeds=[1, 2, 3], batch=64, backbone_name="regnety_032", version="v54",
            desc="RegNetY-3.2GF late-fusion")
finalize(nb, "improvedv54_source.ipynb")

# ---- v55: EfficientNet-B3 late fusion (v47 dual-branch effnet, b0 -> b3) ----
nb = base_v47()
src4 = "".join(nb["cells"][4]["source"]).replace('timm.create_model("efficientnet_b0"',
                                                 'timm.create_model("efficientnet_b3"')
# fix the self-test label so the on-Kaggle abort-check prints the true backbone
src4 = src4.replace('"EfficientNet-B0" if USE_EFFICIENTNET else "ResNet-18"',
                    '"EfficientNet-B3" if USE_EFFICIENTNET else "ResNet-18"')
src4 = src4.replace("'EfficientNet-B0' if USE_EFFICIENTNET else 'ResNet-18'",
                    "'EfficientNet-B3' if USE_EFFICIENTNET else 'ResNet-18'")
nb["cells"][4]["source"] = src4.splitlines(keepends=True)
nb["cells"][0]["source"] = header("v55", "EfficientNet-B3 late-fusion",
    "EfficientNet-B3 (~12M params/branch vs B2's 9.2M, B0's 5.3M) — the most capacity in the batch. "
    "Tests how far the capacity lever goes on the pseudo-expanded 173k-cell set.")
patch_cell2(nb, seeds=[1, 2, 3], batch=64, version="v55", desc="EfficientNet-B3 late-fusion")
finalize(nb, "improvedv55_source.ipynb")

print("Built + validated: improvedv53_source.ipynb (ConvNeXt-early), "
      "improvedv54_source.ipynb (RegNetY-late), improvedv55_source.ipynb (EffNet-B3-late)")
