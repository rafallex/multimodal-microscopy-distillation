"""Locally smoke-test the model-construction cell of each GPU notebook.

Execs cell 4 (model defs + built-in self-test) of v48/v49/v50/v51 with
pretrained=False (no downloads), confirming for each backbone:
  - the dual-branch model builds without error,
  - the fusion head auto-sizes from num_features,
  - forward(zeros(2,1,128,128), zeros(2,1,128,128)) -> shape (2,),
  - the per-branch first conv was adapted to 1 input channel.

This catches a broken backbone swap in seconds, locally, instead of after a
multi-hour GPU run. (pretrained=False skips the ImageNet weight download AND the
weight-copy branch; we separately assert the 1-ch conv shape, which is what the
copy targets.)
"""
import sys, json
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

ROOT = Path(__file__).resolve().parent.parent.parent
NB = ROOT / "notebooks"

# (file, USE_EFFICIENTNET, BACKBONE_NAME or None, expected feature-dim hint)
CASES = [
    ("improvedv48_source.ipynb", True,  None,            "EfficientNet-B2"),
    ("improvedv49_source.ipynb", False, None,            "ResNet-50"),
    ("improvedv50_source.ipynb", True,  None,            "EfficientNet-B0"),
    ("improvedv51_source.ipynb", True,  "convnext_tiny", "ConvNeXt-Tiny"),
]


def first_conv_in_channels(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m.in_channels
    return None


def run(fname, use_eff, backbone_name, label):
    src4 = "".join(json.loads((NB / fname).read_text(encoding="utf-8"))["cells"][4]["source"])
    # Exec the FULL cell (its own self-test builds + forwards the model with
    # pretrained=False). Cutting is unsafe because _make_timm_branch itself
    # contains a `with torch.no_grad():` block.
    defs = src4

    ns = {"torch": torch, "nn": nn, "models": models,
          "USE_EFFICIENTNET": use_eff, "DROPOUT": 0.4}
    if backbone_name:
        ns["BACKBONE_NAME"] = backbone_name
    try:
        exec(defs, ns)
        model = ns["MultimodalClassifier"](pretrained=False).eval()
        x = torch.zeros(2, 1, 128, 128)
        with torch.no_grad():
            out = model(x, x)
        n_params = sum(p.numel() for p in model.parameters())
        bf_in = first_conv_in_channels(model.bf_branch)
        ok_shape = tuple(out.shape) == (2,)
        ok_1ch = bf_in == 1
        status = "PASS" if (ok_shape and ok_1ch) else "FAIL"
        print(f"  [{status}] {label:<16} out={tuple(out.shape)}  params={n_params/1e6:5.1f}M  "
              f"branch_in_ch={bf_in}  ({fname})")
        return ok_shape and ok_1ch
    except Exception as e:
        print(f"  [ERROR] {label:<16} {type(e).__name__}: {e}  ({fname})")
        return False


print("Smoke-testing model construction (pretrained=False, CPU):\n")
results = [run(*c) for c in CASES]
print()
if all(results):
    print(f"All {len(results)} backbones build + forward correctly. Safe to run on Kaggle.")
    sys.exit(0)
print(f"{results.count(False)} of {len(results)} FAILED — fix before any GPU run.")
sys.exit(1)
