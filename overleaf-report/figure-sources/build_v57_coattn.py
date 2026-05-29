"""Builder for v57 = CAFNet-style INTERMEDIATE CO-ATTENTION FUSION.

The Lian/Lindblad/Sladoje 2024 paper (the source of this dataset) shows intermediate
co-attention fusion (CAFNet, ResNet-50) BEATS early (v52) and late (v47/v49) fusion.
The fleet has never tried it. v57 swaps v49's late-concat head for a cross-attention
fusion of the two ResNet-50 feature maps -- an interface-preserving change
(forward(bf,fl) -> [B] logits), so the proven pipeline (soft-pseudo, SWA, AdaBN, TTA,
MIL aux) is untouched. Also distils the L1 ensemble teacher (seed-2 fallback).

Architecture CPU-smoke-tested (forward + backward + learns). Emits
notebooks/improvedv57_source.ipynb. Untested on GPU by design.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "notebooks" / "improvedv49_source.ipynb"
OUT = ROOT / "notebooks" / "improvedv57_source.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))

MODEL_CELL = r'''# === v57: CAFNet-style INTERMEDIATE CO-ATTENTION FUSION (ResNet-50 dual branch) ===
# Replaces v49's late-concat head. Each modality's tokens cross-attend to the OTHER
# modality (the paper's co-attention), fusing intermediate ResNet-50 feature maps.
# forward(bf, fl) -> [B] logits  (same contract as v49: MIL aux reuses these logits).

def _make_featmap_branch(pretrained=True):
    net = models.resnet50(weights="DEFAULT" if pretrained else None)
    w = net.conv1.weight.data
    conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        conv.weight.data = w.mean(dim=1, keepdim=True)   # avg RGB -> 1ch
    net.conv1 = conv
    fd = net.fc.in_features                                # 2048
    feat = nn.Sequential(*list(net.children())[:-2])       # drop avgpool+fc -> [B,2048,H,W]
    return feat, fd


class CoAttnFusion(nn.Module):
    """Each modality attends to the other (cross-attention), then mean-pool tokens."""
    def __init__(self, in_dim, dim=512, heads=8, p=0.1):
        super().__init__()
        self.proj_bf = nn.Conv2d(in_dim, dim, 1)
        self.proj_fl = nn.Conv2d(in_dim, dim, 1)
        self.a_bf = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)
        self.a_fl = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)
        self.n_bf = nn.LayerNorm(dim)
        self.n_fl = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, mbf, mfl):
        mbf, mfl = self.proj_bf(mbf), self.proj_fl(mfl)      # [B,dim,H,W]
        tbf = mbf.flatten(2).transpose(1, 2)                 # [B,HW,dim]
        tfl = mfl.flatten(2).transpose(1, 2)
        obf, _ = self.a_bf(tbf, tfl, tfl)                    # BF queries FL
        ofl, _ = self.a_fl(tfl, tbf, tbf)                    # FL queries BF
        tbf = self.n_bf(tbf + obf)
        tfl = self.n_fl(tfl + ofl)
        return tbf.mean(1), tfl.mean(1)                      # [B,dim] each


class MultimodalClassifier(nn.Module):
    """v57: dual ResNet-50 + intermediate co-attention fusion (CAFNet-style)."""
    def __init__(self, pretrained=True, dropout=DROPOUT):
        super().__init__()
        self.bf_branch, fd = _make_featmap_branch(pretrained)
        self.fl_branch, _ = _make_featmap_branch(pretrained)
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
    print("Backbone: ResNet-50 dual + intermediate co-attention fusion (CAFNet-style)")
    del _m, _x
'''

# --- locate + replace the model cell ---
mi = next(i for i, c in enumerate(nb["cells"])
          if c["cell_type"] == "code" and "class MultimodalClassifier" in "".join(c["source"]))
# safety: ensure the old branch-helper names aren't referenced in OTHER cells
for i, c in enumerate(nb["cells"]):
    if i == mi or c["cell_type"] != "code":
        continue
    body = "".join(c["source"])
    for nm in ("_make_branch(", "_make_effnet_b0_branch(", "_make_resnet18_branch("):
        if nm in body:
            raise SystemExit(f"[FAIL] {nm} referenced in cell {i}; wholesale replace unsafe")
nb["cells"][mi]["source"] = MODEL_CELL.splitlines(keepends=True)
print(f"replaced model cell {mi}")


def edit_cell_contains(substr_locator, old, new, required=True):
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if substr_locator in s and old in s:
            c["source"] = s.replace(old, new).splitlines(keepends=True)
            return True
    if required:
        raise SystemExit(f"[FAIL] could not edit: {old!r}")
    return False


# L1: prepend ensemble-teacher candidates (seed-2 fallback kept)
edit_cell_contains('_PSEUDO_LABEL_CANDIDATES',
    '_PSEUDO_LABEL_CANDIDATES = [\n    "/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv",',
    '_PSEUDO_LABEL_CANDIDATES = [\n'
    '    "/kaggle/input/datasets/rafaelproena/teacherv47ensemble/teacher_v47seeds_mean.csv",\n'
    '    "/kaggle/input/teacherv47ensemble/teacher_v47seeds_mean.csv",\n'
    '    "/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv",')

# smaller batch for the heavier co-attention model on a T4
edit_cell_contains("BATCH_SIZE", "BATCH_SIZE  = 64", "BATCH_SIZE  = 48", required=False)

# teacher metadata label (cosmetic)
edit_cell_contains('"teacher":', '"teacher": "v46_ensemble_lb0.8236"',
                   '"teacher": "v47_3seed_ensemble_mean"', required=False)

# --- markdown title ---
nb["cells"][0]["source"] = (
    "# Multimodal Cancer Challenge 2026 — v57: CAFNet-style intermediate CO-ATTENTION fusion\n"
    "\n"
    "**The SOTA architecture for this exact dataset.** Lian, Lindblad & Sladoje (2024) — the\n"
    "paper this competition's data comes from — show **intermediate co-attention fusion\n"
    "(CAFNet, ResNet-50) beats early and late fusion** (their F1 83.34%). Your fleet has tried\n"
    "late (v47/v49) and early (v52) fusion, but never this.\n"
    "\n"
    "v57 = v49's proven pipeline (soft-pseudo distillation, SWA, AdaBN, 40-way TTA, MIL aux)\n"
    "with the late-concat head replaced by **cross-attention between the two ResNet-50 feature\n"
    "maps** (each modality attends to the other). Interface-preserving: `forward(bf,fl)->[B]`.\n"
    "Also distils the **L1 ensemble teacher** (3-seed mean; seed-2 auto-fallback).\n"
    "\n"
    "Architecture CPU-smoke-tested (forward+backward+learns). **Untested on GPU** — your best\n"
    "shot at a genuinely new, SOTA-grounded high model. Upload `teacher_v47seeds_mean.csv` as\n"
    "Kaggle dataset `teacherv47ensemble` (expect load log mean ~0.49).\n"
).splitlines(keepends=True)

# normalize all sources to list form
for c in nb["cells"]:
    c["source"] = "".join(c["source"]).splitlines(keepends=True)

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"WROTE {OUT.name}")

# --- validate: compile (skip magics) + key pieces present ---
re_nb = json.loads(OUT.read_text(encoding="utf-8"))
for i, c in enumerate(re_nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        pysrc = "\n".join(ln for ln in src.split("\n") if not ln.lstrip().startswith(("!", "%")))
        compile(pysrc, f"cell{i}", "exec")
joined = "\n".join("".join(c["source"]) for c in re_nb["cells"])
assert "class CoAttnFusion" in joined and "MultiheadAttention" in joined
assert "models.resnet50" in joined
assert "teacher_v47seeds_mean.csv" in joined
assert joined.count("class MultimodalClassifier") == 1
print("VALID: co-attention model present, compiles, single classifier, L1 teacher wired")
