"""Builder for v58 = EfficientNet-B0 dual branch + INTERMEDIATE CO-ATTENTION FUSION.

The best of both: the proven backbone (EffNet-B0, LB 0.8355 in late fusion) + the SOTA
fusion mechanism (CAFNet co-attention). ResNet-50 (v49/v57) is weak on this data
(LB 0.79, rank-corr 0.90), so we apply co-attention to EfficientNet instead. Base:
v50 (dual EffNet-B0). Only the fusion changes (late-concat -> co-attention); the proven
pipeline (soft-pseudo, SWA, AdaBN, 40-way TTA, MIL aux) is untouched. Distils the L1
ensemble teacher (seed-2 fallback). Hardened for AMP (fp32 attn + pre-norm + ReZero).

CPU-smoke-tested: 12.0M params, forward (B,), backward, learns, gates engage, finite.
Emits notebooks/improvedv58_source.ipynb. Untested on GPU by design.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "notebooks" / "improvedv50_sourceDO_NOT_RUN_.ipynb"
OUT = ROOT / "notebooks" / "improvedv58_source.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))

MODEL_CELL = r'''# === v58: EfficientNet-B0 dual branch + INTERMEDIATE CO-ATTENTION FUSION ===
# Proven backbone (EffNet-B0, LB 0.8355 in late fusion) + the SOTA fusion mechanism
# (CAFNet co-attention). Replaces v50's late-concat head. forward(bf,fl) -> [B] logits
# (same contract as v50: MIL aux reuses these logits). Hardened for AMP.

def _make_effnet_featmap_branch(pretrained=True):
    import timm
    net = timm.create_model("efficientnet_b0", pretrained=pretrained,
                            num_classes=0, global_pool="")     # global_pool="" -> feature MAP
    old = net.conv_stem
    conv = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                     stride=old.stride, padding=old.padding, bias=False)
    if pretrained:
        conv.weight.data = old.weight.data.mean(dim=1, keepdim=True)  # avg RGB -> 1ch
    net.conv_stem = conv
    return net, net.num_features                                # 1280


class CoAttnFusion(nn.Module):
    """CAFNet-style cross-attention, HARDENED for mixed precision: fp32 attention +
    pre-norm + ReZero gate (alpha init 0 -> attention starts as a no-op and ramps in).
    Prevents the fp16 attention-softmax overflow that NaNed v57 v1."""
    def __init__(self, in_dim, dim=512, heads=8, p=0.1):
        super().__init__()
        self.proj_bf = nn.Conv2d(in_dim, dim, 1)
        self.proj_fl = nn.Conv2d(in_dim, dim, 1)
        self.n_bf = nn.LayerNorm(dim)
        self.n_fl = nn.LayerNorm(dim)
        self.a_bf = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)
        self.a_fl = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)
        self.alpha_bf = nn.Parameter(torch.zeros(1))            # ReZero gates
        self.alpha_fl = nn.Parameter(torch.zeros(1))
        self.out_dim = dim

    def forward(self, mbf, mfl):
        tbf = self.proj_bf(mbf).flatten(2).transpose(1, 2)      # [B,HW,dim]
        tfl = self.proj_fl(mfl).flatten(2).transpose(1, 2)
        with torch.autocast(device_type=tbf.device.type, enabled=False):
            tbf, tfl = tbf.float(), tfl.float()
            qbf, qfl = self.n_bf(tbf), self.n_fl(tfl)           # pre-norm
            obf, _ = self.a_bf(qbf, qfl, qfl)                   # BF queries FL
            ofl, _ = self.a_fl(qfl, qbf, qbf)                   # FL queries BF
            tbf = tbf + self.alpha_bf * obf                     # ReZero residual
            tfl = tfl + self.alpha_fl * ofl
        return tbf.mean(1), tfl.mean(1)                         # [B,dim] each


class MultimodalClassifier(nn.Module):
    """v58: dual EfficientNet-B0 + intermediate co-attention fusion (CAFNet-style)."""
    def __init__(self, pretrained=True, dropout=DROPOUT):
        super().__init__()
        self.bf_branch, fd = _make_effnet_featmap_branch(pretrained)
        self.fl_branch, _ = _make_effnet_featmap_branch(pretrained)
        self.fusion = CoAttnFusion(fd, dim=512, heads=8, p=0.1)
        d = self.fusion.out_dim
        self.head = nn.Sequential(
            nn.Linear(d * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, bf, fl):
        vbf, vfl = self.fusion(self.bf_branch(bf), self.fl_branch(fl))
        return self.head(torch.cat([vbf, vfl], dim=1)).squeeze(-1)


with torch.no_grad():
    _m = MultimodalClassifier(pretrained=False).cpu()
    _x = torch.zeros(2, 1, 128, 128)
    n_params = sum(p.numel() for p in _m.parameters())
    print(f"Output shape: {_m(_x, _x).shape}   params: {n_params / 1e6:.1f}M")
    print("Backbone: EfficientNet-B0 dual + intermediate co-attention fusion (CAFNet-style)")
    del _m, _x
'''

mi = next(i for i, c in enumerate(nb["cells"])
          if c["cell_type"] == "code" and "class MultimodalClassifier" in "".join(c["source"]))
for i, c in enumerate(nb["cells"]):
    if i == mi or c["cell_type"] != "code":
        continue
    body = "".join(c["source"])
    for nm in ("_make_branch(", "_make_effnet_b0_branch(", "_make_resnet18_branch("):
        if nm in body:
            raise SystemExit(f"[FAIL] {nm} referenced in cell {i}; wholesale replace unsafe")
nb["cells"][mi]["source"] = MODEL_CELL.splitlines(keepends=True)
print(f"replaced model cell {mi}")


def edit_contains(loc, old, new, required=True):
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if loc in s and old in s:
            c["source"] = s.replace(old, new).splitlines(keepends=True)
            return True
    if required:
        raise SystemExit(f"[FAIL] could not edit: {old!r}")
    return False


# seeds 401-404 (lottery) -> [1, 2]
edit_contains("SEEDS", "SEEDS               = [401, 402, 403, 404]", "SEEDS               = [1, 2]", required=False)
# co-attention is a bit heavier than late-concat -> trim batch for T4 headroom
edit_contains("BATCH_SIZE", "BATCH_SIZE  = 128", "BATCH_SIZE  = 96", required=False)
# L1 ensemble teacher (seed-2 fallback)
edit_contains('_PSEUDO_LABEL_CANDIDATES',
    '_PSEUDO_LABEL_CANDIDATES = [\n    "/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv",',
    '_PSEUDO_LABEL_CANDIDATES = [\n'
    '    "/kaggle/input/datasets/rafaelproena/teacherv47ensemble/teacher_v47seeds_mean.csv",\n'
    '    "/kaggle/input/teacherv47ensemble/teacher_v47seeds_mean.csv",\n'
    '    "/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv",')
edit_contains('"teacher":', '"teacher": "v46_ensemble_lb0.8236"', '"teacher": "v47_3seed_ensemble_mean"', required=False)

nb["cells"][0]["source"] = (
    "# Multimodal Cancer Challenge 2026 — v58: EfficientNet-B0 + intermediate CO-ATTENTION fusion\n"
    "\n"
    "**Proven backbone + SOTA fusion.** ResNet-50 (v49/v57) is weak on this data (LB 0.79,\n"
    "rank-corr 0.90 with v47_s2), so we keep the backbone that actually works here\n"
    "(EfficientNet-B0, LB 0.8355 in late fusion) and swap in the fusion the source-dataset\n"
    "authors report as state of the art: **intermediate co-attention** (CAFNet) -- each\n"
    "modality's feature map cross-attends to the other before pooling.\n"
    "\n"
    "Only the fusion changes vs the proven recipe (late-concat -> co-attention). Hardened for\n"
    "AMP (fp32 attention + pre-norm + ReZero gate, so it starts as stable as the bare\n"
    "backbone). Distils the **L1 ensemble teacher** (3-seed mean; seed-2 auto-fallback).\n"
    "12M params, CPU-smoke-tested. **This is the strongest single-model bet** -- run it over\n"
    "the ResNet co-attention v57.\n"
).splitlines(keepends=True)

for c in nb["cells"]:
    c["source"] = "".join(c["source"]).splitlines(keepends=True)

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"WROTE {OUT.name}")

re_nb = json.loads(OUT.read_text(encoding="utf-8"))
for i, c in enumerate(re_nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        pysrc = "\n".join(ln for ln in src.split("\n") if not ln.lstrip().startswith(("!", "%")))
        compile(pysrc, f"cell{i}", "exec")
joined = "\n".join("".join(c["source"]) for c in re_nb["cells"])
assert "class CoAttnFusion" in joined and "alpha_bf" in joined
assert '_make_effnet_featmap_branch' in joined and 'global_pool=""' in joined
assert joined.count("class MultimodalClassifier") == 1
assert "teacher_v47seeds_mean.csv" in joined
print("VALID: EffNet co-attention model present, compiles, single classifier, L1 teacher wired")
