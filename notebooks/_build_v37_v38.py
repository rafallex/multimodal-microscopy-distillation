"""Build v37 (modality-specific aug) and v38 (early fusion) source notebooks by
forking v19 and surgically replacing specific cells.

Run from anywhere; output lands in the notebooks/ directory next to this script.

v37: only the augmentation pipeline changes (cell 6). Everything else mirrors v19.
v38: only the model architecture changes (cell 4) — early fusion (2-channel input,
     single backbone). Augmentation and training are v19 verbatim.

Source: Lian et al. 2024 "Let it shine" (CBM 185:109498), §5.2 + Tables 2-3.
"""
from pathlib import Path
import json
import copy

HERE = Path(__file__).parent.resolve()
SRC = HERE / "improvedv19_source.ipynb"
V37 = HERE / "improvedv37_source.ipynb"
V38 = HERE / "improvedv38_source.ipynb"


def load_v19():
    with open(SRC, encoding="utf-8") as f:
        return json.load(f)


def src_lines(text: str):
    """Convert a plain string into the per-line list format used in .ipynb cells."""
    lines = text.split("\n")
    # Each line except the last needs a trailing \n; the last has no trailing \n.
    out = [l + "\n" for l in lines[:-1]]
    if lines[-1] != "":
        out.append(lines[-1])
    return out


# ============================================================================
# v37 — modality-specific augmentation
# ============================================================================
V37_TITLE_MD = r"""# Multimodal Cancer Classification Challenge 2026 — v37 (Lian-aligned aug)

**Single-variable change from v19: replace the shared per-modality transform with the modality-specific augmentation pipeline from Lian et al. 2024 ("Let it shine", CBM 185:109498), §5.2.**

## Motivation

Lian et al. is the published SOTA on this exact dataset (the paper is from the same group that runs the Kaggle competition; the lead author of [11] is our course instructor). Their CAFNet model reaches **F1=0.8334 / Accuracy=0.9179 / ROC AUC=0.9686** under 3-fold CV.

Their ablation (Table 2) shows the single highest-impact augmentation choice:

| FL augmentation | F1 |
|---|---|
| FL-only **with** color jitter | **0.7399** |
| FL-only **without** color jitter | 0.5514 |

**Removing color jitter from the FL branch costs ~19 percentage points of F1.** For the BF branch, the same ablation only costs ~2 pp. The mechanism is intuitive: FL intensity is far more session/instrument-variable than BF (autofluorescence depends on excitation power, filter wear, scope drift), so color invariance on FL has to be learned aggressively.

Our v19 used a **shared** transform across both modalities — `ColorJitter(brightness=0.4, contrast=0.4)` on both — which is mild on BF (probably fine) and almost certainly too weak on FL.

## What changes vs v19

| Component | v19 | **v37** | Source |
|---|---|---|---|
| **BF aug** | ColorJitter(0.4, 0.4) | RandomChoice of {posterize bits=3 p=0.4, blur k=5 σ=1.5 p=0.2, solarize thr=100 p=0.4} → ColorJitter(0.5, 0.2) | Lian §5.2 BF |
| **FL aug** | ColorJitter(0.4, 0.4) | GaussianBlur(k=5, σ∈[0.3, 3.2]) → ColorJitter(**0.8, 0.8**) | Lian §5.2 FL |
| Paired geo aug | D4 + ±10° + ±15° affine | unchanged | — |
| Model | EffNet-B0 dual-branch late fusion | unchanged | — |
| Loss | BCE + MIL (w=0.5) | unchanged | — |
| Epochs / LR / opt | 12 / 3e-4 / AdamW OneCycle | unchanged | — |
| TTA / AdaBN / stain norm | 8-way D4 / on / on | unchanged | — |

## Expected outcome

If the Lian ablation transfers to our competition data, v37 should clear v19's LB 0.7455 by a noticeable margin (rough estimate +0.01 to +0.04 LB AUC). If it doesn't move, that tells us their FL color-jitter finding doesn't replicate at 128×128 grayscale (their FL is 256×256 with 4 emission channels — more variance to learn from).

## Note on 1-channel FL

Lian's FL is 4-channel (emission wavelengths 465/517/568/668). Ours is 1-channel grayscale. `T.ColorJitter(brightness, contrast)` works on any channel count, so the recipe transfers. The `hue`/`saturation` parameters from Lian's BF recipe only make sense on multi-channel input; we drop them.
"""

# Replacement code for cell 6 — modality-specific transforms
V37_CELL6_CODE = r'''import torchvision.transforms as T

def _to_tensor_norm(mean, std):
    def fn(img):
        t = TF.to_tensor(img)
        return TF.normalize(t, [mean], [std])
    return fn

to_tensor_bf = _to_tensor_norm(BF_MEAN, BF_STD)
to_tensor_fl = _to_tensor_norm(FL_MEAN, FL_STD)

class PairedGeoAug:
    """D4 + small rotation + paired affine — applied identically to BF and FL.

    Identical to v19. Geometric augs must remain paired so BF/FL stay registered.
    """
    def __init__(self, p_hflip=0.5, p_vflip=0.5, rot90=True, max_rot=10.0,
                 affine_deg=0.0, affine_translate=0.0):
        self.p_hflip = p_hflip; self.p_vflip = p_vflip
        self.rot90 = rot90; self.max_rot = max_rot
        self.affine_deg = affine_deg; self.affine_translate = affine_translate
    def __call__(self, bf, fl):
        if self.rot90:
            k = random.randint(0, 3)
            if k:
                bf = torch.rot90(bf, k, dims=(-2, -1))
                fl = torch.rot90(fl, k, dims=(-2, -1))
        if random.random() < self.p_hflip: bf, fl = TF.hflip(bf), TF.hflip(fl)
        if random.random() < self.p_vflip: bf, fl = TF.vflip(bf), TF.vflip(fl)
        if self.max_rot > 0:
            a = random.uniform(-self.max_rot, self.max_rot)
            bf, fl = TF.rotate(bf, a), TF.rotate(fl, a)
        if self.affine_deg > 0 or self.affine_translate > 0:
            H, W = bf.shape[-2], bf.shape[-1]
            angle = random.uniform(-self.affine_deg, self.affine_deg) if self.affine_deg > 0 else 0.0
            tx = random.uniform(-self.affine_translate, self.affine_translate) * W if self.affine_translate > 0 else 0
            ty = random.uniform(-self.affine_translate, self.affine_translate) * H if self.affine_translate > 0 else 0
            bf = TF.affine(bf, angle=angle, translate=(int(tx), int(ty)), scale=1.0, shear=0.0)
            fl = TF.affine(fl, angle=angle, translate=(int(tx), int(ty)), scale=1.0, shear=0.0)
        return bf, fl


# === v37: modality-specific augmentation (Lian 2024 §5.2) ===
#
# BF branch: a SINGLE pixel-level corruption is picked per image — posterize (3-bit),
# Gaussian blur (k=5 σ=1.5), or solarize (threshold=100) — with probabilities
# 0.4 / 0.2 / 0.4 (which sum to 1.0, i.e. one of the three always applies).
# After that, ColorJitter(brightness=0.5, contrast=0.2). Lian's recipe also adds
# saturation=0.2 hue=0.2, but those only make sense on multi-channel input; our BF
# is grayscale, so we drop them.
#
# FL branch: GaussianBlur with σ randomly sampled from [0.3, 3.2] (essentially
# "light-to-heavy blur"), followed by ColorJitter(brightness=0.8, contrast=0.8) —
# the heavier 0.8 magnitude is Lian's key finding (their ablation shows this single
# choice is worth ~19 pp F1).
#
# Both pipelines act on PIL "L"-mode (uint8 grayscale) input, then to_tensor +
# normalize at the end.

def train_modality_transform(modality):
    if modality == "bf":
        return T.Compose([
            T.RandomChoice(
                [
                    T.RandomPosterize(bits=3, p=1.0),
                    T.GaussianBlur(kernel_size=5, sigma=(1.5, 1.5)),
                    T.RandomSolarize(threshold=100, p=1.0),
                ],
                p=[0.4, 0.2, 0.4],
            ),
            T.ColorJitter(brightness=0.5, contrast=0.2),
            to_tensor_bf,
        ])
    elif modality == "fl":
        return T.Compose([
            T.GaussianBlur(kernel_size=5, sigma=(0.3, 3.2)),
            T.ColorJitter(brightness=0.8, contrast=0.8),
            to_tensor_fl,
        ])
    raise ValueError(modality)


def eval_modality_transform(modality):
    return to_tensor_bf if modality == "bf" else to_tensor_fl


def build_paired_aug():
    """v19 paired aug — unchanged in v37."""
    return PairedGeoAug(max_rot=10.0, affine_deg=15.0, affine_translate=0.10)


print("Augmentation summary (v37 — Lian-aligned modality-specific):")
print("  BF:  RandomChoice{posterize3b p=0.4, blur σ=1.5 p=0.2, solarize128 p=0.4}")
print("       -> ColorJitter(brightness=0.5, contrast=0.2)")
print("  FL:  GaussianBlur(σ∈[0.3, 3.2])")
print("       -> ColorJitter(brightness=0.8, contrast=0.8)  <-- key change vs v19")
print("  Geom (paired): D4 + ±10° rot + ±15° affine + 10% translate")
'''


def build_v37():
    nb = load_v19()
    nb["cells"][0]["source"] = src_lines(V37_TITLE_MD)
    nb["cells"][6]["source"] = src_lines(V37_CELL6_CODE)
    # Clear any cell outputs/execution counts to keep _source.ipynb clean
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    with open(V37, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Wrote {V37.name}")


# ============================================================================
# v38 — early fusion (concat BF+FL at input, single backbone)
# ============================================================================
V38_TITLE_MD = r"""# Multimodal Cancer Classification Challenge 2026 — v38 (early fusion)

**Single-variable change from v19: switch from late fusion (two backbones + concat features) to early fusion (concat BF+FL at input + single backbone).**

## Motivation

Lian et al. 2024 ("Let it shine", CBM 185:109498) Table 3 ranks the fusion strategies on our exact task:

| Fusion strategy | F1 (3-fold CV) | ROC AUC |
|---|---|---|
| BF-only | 0.6944 | 0.8958 |
| FL-only | 0.7399 | 0.9039 |
| **Late fusion** (our v19) | 0.8104 | 0.9495 |
| **Early fusion** | **0.8273** | **0.9626** |
| MMTM (intermediate) | 0.8151 | 0.9556 |
| HcCNN (intermediate) | 0.8243 | 0.9591 |
| CAFNet (intermediate) | 0.8334 | 0.9686 |

**Early fusion beats late fusion by 1.69 pp F1 / 1.31 pp ROC AUC**, and gets within 0.61 pp of CAFNet (the SOTA they report) at a tiny fraction of the implementation cost. Their discussion (§7) explicitly recommends early fusion as the best cost/performance trade-off: "_early fusion, requiring only about one-third of the training time of CAFNet, achieves comparable results and can be a recommended approach to balance the advantages of multimodal information with the required resources for its processing._"

The mechanism: at the input layer, the network can learn pixel-level cross-modal correspondences (e.g. "this is a region where BF is dim AND FL is bright" — a malignant-cell signature in their data). Late fusion can never see this — by the time the two streams meet at the head, all pixel-level joint information has been compressed away.

## What changes vs v19

| Component | v19 (late fusion) | **v38 (early fusion)** |
|---|---|---|
| Model.bf_branch | EffNet-B0 1-ch input | — |
| Model.fl_branch | EffNet-B0 1-ch input | — |
| **Model.backbone** | (two of the above) | **EffNet-B0 with 2-ch input (BF + FL concatenated channel-wise)** |
| Model.head | Linear(2·fd → hidden → 1) | **Linear(fd → hidden → 1)** |
| Param count | ~2·M | ~M (roughly half) |
| Forward pass | `concat([bf_branch(bf), fl_branch(fl)], dim=1)` | `backbone(concat([bf, fl], dim=1))` |
| Augmentation | v19 (shared per-modality) | unchanged from v19 |
| Loss / optimizer / TTA | v19 | unchanged |

## Caveats

- **Alignment-sensitive.** Lian Fig. 11 shows early fusion degrades significantly when BF and FL are shifted by 8-16 px. The Kaggle test set's BF/FL alignment quality is unknown to us — if it's worse than the training set, late fusion may still win on the LB even though early fusion wins on CV.
- **Won't combine trivially with v37.** v37 changes the augmentation pipeline; v38 changes the architecture. If both individually improve, v39 should stack them. If they don't compose additively, that's informative too.
- **TTA still applies normally** — the 8-way D4 just acts on the 2-channel input.

## Expected outcome

If Lian's late-vs-early gap (1.69 pp F1) transfers to our LB AUC, we'd see roughly +0.01 to +0.02 LB AUC vs v19's 0.7455. Less than v37's expected lift, but a much cleaner experiment because exactly one thing changes.
"""

V38_CELL4_CODE = r'''def _make_resnet18_branch(pretrained=True, in_channels=1):
    """Single-backbone ResNet-18 with configurable input channels."""
    weights = "DEFAULT" if pretrained else None
    net = models.resnet18(weights=weights)
    w = net.conv1.weight.data  # [64, 3, 7, 7]
    new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        # For 1-ch: average RGB filters. For 2-ch: tile the averaged-RGB filter
        # across both input channels so each modality starts from the same pretrained init.
        mean_rgb = w.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
        new_conv.weight.data = mean_rgb.repeat(1, in_channels, 1, 1) / in_channels
    net.conv1 = new_conv
    fd = net.fc.in_features; net.fc = nn.Identity()
    return net, fd

def _make_effnet_b0_branch(pretrained=True, in_channels=1):
    import timm
    net = timm.create_model("efficientnet_b0", pretrained=pretrained,
                            num_classes=0, global_pool="avg")
    old = net.conv_stem  # [32, 3, 3, 3]
    new_conv = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                         stride=old.stride, padding=old.padding, bias=False)
    if pretrained:
        mean_rgb = old.weight.data.mean(dim=1, keepdim=True)  # [32, 1, 3, 3]
        new_conv.weight.data = mean_rgb.repeat(1, in_channels, 1, 1) / in_channels
    net.conv_stem = new_conv
    return net, net.num_features  # 1280

def _make_backbone(pretrained=True, in_channels=1):
    if USE_EFFICIENTNET:
        return _make_effnet_b0_branch(pretrained, in_channels=in_channels)
    return _make_resnet18_branch(pretrained, in_channels=in_channels)


# === v38: EARLY-FUSION single-backbone classifier ===
#
# Architecture: BF and FL are concatenated along the channel dimension to form a
# 2-channel input, then passed through a single EfficientNet-B0 (or ResNet-18) whose
# first conv has been re-initialized to accept 2 channels. The head is a small MLP
# on the backbone's feature vector.
#
# Pretraining: the new 2-channel conv1 weights are initialized by averaging the
# pretrained RGB filters into one channel, then tiling that channel-mean across
# both input channels. Dividing by in_channels keeps the output magnitude roughly
# matched to the original 3-channel input, which preserves the conditioning of
# subsequent layers (BatchNorm stats etc.).
class MultimodalClassifier(nn.Module):
    def __init__(self, pretrained=True, dropout=DROPOUT):
        super().__init__()
        self.backbone, fd = _make_backbone(pretrained, in_channels=2)
        hidden = 512 if fd >= 512 else 256
        self.head = nn.Sequential(
            nn.Linear(fd, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, bf, fl):
        # bf, fl: [B, 1, H, W] each. Concat to [B, 2, H, W].
        x = torch.cat([bf, fl], dim=1)
        return self.head(self.backbone(x)).squeeze(-1)

with torch.no_grad():
    _m = MultimodalClassifier(pretrained=False).cpu()
    _x = torch.zeros(2, 1, 128, 128)
    n_params = sum(p.numel() for p in _m.parameters())
    print(f"Output shape: {_m(_x, _x).shape}   params: {n_params / 1e6:.1f}M")
    print(f"Backbone: {'EfficientNet-B0' if USE_EFFICIENTNET else 'ResNet-18'} (early-fusion, 2-ch input)")
    del _m, _x
'''


def build_v38():
    nb = load_v19()
    nb["cells"][0]["source"] = src_lines(V38_TITLE_MD)
    nb["cells"][4]["source"] = src_lines(V38_CELL4_CODE)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    with open(V38, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Wrote {V38.name}")


if __name__ == "__main__":
    build_v37()
    build_v38()
