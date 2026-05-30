"""CPU smoke-test: EfficientNet-B0 dual branch + hardened co-attention fusion (v58).
Proven backbone (EffNet, 0.8355 in late fusion) + SOTA fusion (co-attention).
Validates EffNet feature-map extraction, forward [B], backward, learning, finite output.
"""
import torch, torch.nn as nn, timm

DROPOUT = 0.4


def _make_effnet_featmap_branch(pretrained=False):
    net = timm.create_model("efficientnet_b0", pretrained=pretrained,
                            num_classes=0, global_pool="")   # global_pool="" -> feature MAP
    old = net.conv_stem
    conv = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                     stride=old.stride, padding=old.padding, bias=False)
    if pretrained:
        conv.weight.data = old.weight.data.mean(dim=1, keepdim=True)
    net.conv_stem = conv
    return net, net.num_features          # 1280


class CoAttnFusion(nn.Module):
    def __init__(self, in_dim, dim=512, heads=8, p=0.1):
        super().__init__()
        self.proj_bf = nn.Conv2d(in_dim, dim, 1)
        self.proj_fl = nn.Conv2d(in_dim, dim, 1)
        self.n_bf = nn.LayerNorm(dim)
        self.n_fl = nn.LayerNorm(dim)
        self.a_bf = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)
        self.a_fl = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)
        self.alpha_bf = nn.Parameter(torch.zeros(1))
        self.alpha_fl = nn.Parameter(torch.zeros(1))
        self.out_dim = dim

    def forward(self, mbf, mfl):
        tbf = self.proj_bf(mbf).flatten(2).transpose(1, 2)
        tfl = self.proj_fl(mfl).flatten(2).transpose(1, 2)
        with torch.autocast(device_type=tbf.device.type, enabled=False):
            tbf, tfl = tbf.float(), tfl.float()
            qbf, qfl = self.n_bf(tbf), self.n_fl(tfl)
            obf, _ = self.a_bf(qbf, qfl, qfl)
            ofl, _ = self.a_fl(qfl, qbf, qbf)
            tbf = tbf + self.alpha_bf * obf
            tfl = tfl + self.alpha_fl * ofl
        return tbf.mean(1), tfl.mean(1)


class MultimodalClassifier(nn.Module):
    def __init__(self, pretrained=True, dropout=DROPOUT):
        super().__init__()
        self.bf_branch, fd = _make_effnet_featmap_branch(pretrained)
        self.fl_branch, _ = _make_effnet_featmap_branch(pretrained)
        self.fusion = CoAttnFusion(fd, dim=512, heads=8, p=0.1)
        d = self.fusion.out_dim
        self.head = nn.Sequential(
            nn.Linear(d * 2, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(512, 1))

    def forward(self, bf, fl):
        vbf, vfl = self.fusion(self.bf_branch(bf), self.fl_branch(fl))
        return self.head(torch.cat([vbf, vfl], dim=1)).squeeze(-1)


torch.manual_seed(0)
m = MultimodalClassifier(pretrained=False)
n = sum(p.numel() for p in m.parameters())
xb, xf = torch.randn(4, 1, 128, 128), torch.randn(4, 1, 128, 128)
with torch.no_grad():
    fm = m.bf_branch(xb)
print(f"EffNet-B0 feature map for 128x128: {tuple(fm.shape)} -> {fm.shape[-1]*fm.shape[-2]} tokens")
out = m(xb, xf)
print(f"forward OK: {tuple(out.shape)} (expect (4,))   params {n/1e6:.1f}M")
assert out.shape == (4,)
y = torch.tensor([1.0, 0.0, 1.0, 0.0]); opt = torch.optim.AdamW(m.parameters(), 1e-3)
lf = nn.BCEWithLogitsLoss(); m.train(); ls = []
for _ in range(25):
    opt.zero_grad(); l = lf(m(xb, xf), y); l.backward(); opt.step(); ls.append(l.item())
print(f"backward OK. loss {ls[0]:.4f} -> {ls[-1]:.4f}  ({'LEARNS' if ls[-1] < ls[0]*0.5 else 'flat?'})")
print(f"  ReZero gates: alpha_bf={m.fusion.alpha_bf.item():+.4f} alpha_fl={m.fusion.alpha_fl.item():+.4f}")
assert torch.isfinite(m(xb, xf)).all()
print("ALL OK -- EffNet-B0 + co-attention is sound")
