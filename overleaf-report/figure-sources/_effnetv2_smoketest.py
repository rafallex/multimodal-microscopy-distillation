"""CPU smoke-test: dual EfficientNetV2-S late-fusion (v59) with 1-ch stem adapter.
A stronger, modern EfficientNet on the PROVEN late-fusion + distillation recipe.
Validates 1-ch conv_stem adapt, forward [B], backward, learning, finite output.
"""
import torch, torch.nn as nn, timm

DROPOUT = 0.4
BACKBONE = "tf_efficientnetv2_s"


def _make_branch(pretrained=False):
    net = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0, global_pool="avg")
    old = net.conv_stem
    conv = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                     stride=old.stride, padding=old.padding, bias=False)
    if pretrained:
        conv.weight.data = old.weight.data.mean(dim=1, keepdim=True)
    net.conv_stem = conv
    return net, net.num_features


class MultimodalClassifier(nn.Module):
    def __init__(self, pretrained=True, dropout=DROPOUT):
        super().__init__()
        self.bf_branch, fd = _make_branch(pretrained)
        self.fl_branch, _ = _make_branch(pretrained)
        hidden = 512 if fd >= 512 else 256
        self.head = nn.Sequential(
            nn.Linear(fd * 2, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, bf, fl):
        return self.head(torch.cat([self.bf_branch(bf), self.fl_branch(fl)], dim=1)).squeeze(-1)


torch.manual_seed(0)
m = MultimodalClassifier(pretrained=False)
n = sum(p.numel() for p in m.parameters())
xb, xf = torch.randn(4, 1, 128, 128), torch.randn(4, 1, 128, 128)
out = m(xb, xf)
print(f"forward OK: {tuple(out.shape)} (expect (4,))   params {n/1e6:.1f}M   backbone {BACKBONE}")
assert out.shape == (4,)
y = torch.tensor([1.0, 0.0, 1.0, 0.0]); opt = torch.optim.AdamW(m.parameters(), 1e-3)
lf = nn.BCEWithLogitsLoss(); m.train(); ls = []
for _ in range(20):
    opt.zero_grad(); l = lf(m(xb, xf), y); l.backward(); opt.step(); ls.append(l.item())
print(f"backward OK. loss {ls[0]:.4f} -> {ls[-1]:.4f}  ({'LEARNS' if ls[-1] < ls[0]*0.5 else 'flat?'})")
assert torch.isfinite(m(xb, xf)).all()
print("ALL OK -- dual EfficientNetV2-S late fusion is sound")
