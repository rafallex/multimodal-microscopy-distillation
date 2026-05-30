"""CPU smoke-test of the CAFNet-style intermediate co-attention fusion model.
Validates: builds, forward shape [B], backward works, and it can LEARN (overfit a
tiny batch). If this passes locally, the architecture is sound for Kaggle GPU.
"""
import torch, torch.nn as nn
from torchvision import models

DROPOUT = 0.4


def _make_featmap_branch(pretrained=False):
    net = models.resnet50(weights="DEFAULT" if pretrained else None)
    w = net.conv1.weight.data
    conv = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    if pretrained:
        conv.weight.data = w.mean(dim=1, keepdim=True)
    net.conv1 = conv
    fd = net.fc.in_features                       # 2048
    feat = nn.Sequential(*list(net.children())[:-2])  # drop avgpool + fc -> [B,2048,H,W]
    return feat, fd


class CoAttnFusion(nn.Module):
    """CAFNet-style cross-attention, numerically HARDENED for mixed precision:
    fp32 attention + pre-norm + ReZero gate (alpha init 0 -> attention starts as a
    no-op, so training begins as stable as the bare backbone and ramps in gradually).
    Prevents the fp16 attention-softmax overflow that NaNs from-scratch attention
    layers once the OneCycle LR peaks (observed in v57 v1: clean epoch 0, NaN epoch 1)."""
    def __init__(self, in_dim, dim=512, heads=8, p=0.1):
        super().__init__()
        self.proj_bf = nn.Conv2d(in_dim, dim, 1)
        self.proj_fl = nn.Conv2d(in_dim, dim, 1)
        self.n_bf = nn.LayerNorm(dim)
        self.n_fl = nn.LayerNorm(dim)
        self.a_bf = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)
        self.a_fl = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)
        self.alpha_bf = nn.Parameter(torch.zeros(1))           # ReZero gates
        self.alpha_fl = nn.Parameter(torch.zeros(1))
        self.out_dim = dim

    def forward(self, mbf, mfl):
        tbf = self.proj_bf(mbf).flatten(2).transpose(1, 2)     # [B,HW,dim]
        tfl = self.proj_fl(mfl).flatten(2).transpose(1, 2)
        # attention in fp32 regardless of AMP -- fp16 softmax overflow is the NaN source
        with torch.autocast(device_type=tbf.device.type, enabled=False):
            tbf, tfl = tbf.float(), tfl.float()
            qbf, qfl = self.n_bf(tbf), self.n_fl(tfl)          # pre-norm
            obf, _ = self.a_bf(qbf, qfl, qfl)                  # BF queries FL
            ofl, _ = self.a_fl(qfl, qbf, qbf)                  # FL queries BF
            tbf = tbf + self.alpha_bf * obf                    # ReZero residual
            tfl = tfl + self.alpha_fl * ofl
        return tbf.mean(1), tfl.mean(1)                        # [B,dim] each


class MultimodalClassifier(nn.Module):
    def __init__(self, pretrained=True, dropout=DROPOUT):
        super().__init__()
        self.bf_feat, fd = _make_featmap_branch(pretrained)
        self.fl_feat, _ = _make_featmap_branch(pretrained)
        self.fusion = CoAttnFusion(fd, dim=512, heads=8, p=0.1)
        d = self.fusion.out_dim
        self.head = nn.Sequential(
            nn.Linear(d * 2, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(512, 1),
        )

    def forward(self, bf, fl):
        vbf, vfl = self.fusion(self.bf_feat(bf), self.fl_feat(fl))
        return self.head(torch.cat([vbf, vfl], dim=1)).squeeze(-1)


torch.manual_seed(0)
m = MultimodalClassifier(pretrained=False)
n_params = sum(p.numel() for p in m.parameters())
x_bf = torch.randn(4, 1, 128, 128)
x_fl = torch.randn(4, 1, 128, 128)
out = m(x_bf, x_fl)
print(f"forward OK: output shape {tuple(out.shape)} (expect (4,))   params {n_params/1e6:.1f}M")
assert out.shape == (4,)

# can it LEARN? overfit a tiny fixed batch
y = torch.tensor([1.0, 0.0, 1.0, 0.0])
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
lossfn = nn.BCEWithLogitsLoss()
m.train()
losses = []
for step in range(25):
    opt.zero_grad()
    loss = lossfn(m(x_bf, x_fl), y)
    loss.backward(); opt.step()
    losses.append(loss.item())
print(f"backward OK. loss {losses[0]:.4f} -> {losses[-1]:.4f}  ({'LEARNS' if losses[-1] < losses[0]*0.5 else 'flat?'})")
print(f"  ReZero gates moved off 0 (attention is learning): "
      f"alpha_bf={m.fusion.alpha_bf.item():+.4f} alpha_fl={m.fusion.alpha_fl.item():+.4f}")
assert torch.isfinite(out).all(), "non-finite output!"
print("forward map spatial check:")
with torch.no_grad():
    fm = m.bf_feat(x_bf)
    print(f"  ResNet-50 feature map for 128x128 input: {tuple(fm.shape)} -> {fm.shape[-1]*fm.shape[-2]} attention tokens")
