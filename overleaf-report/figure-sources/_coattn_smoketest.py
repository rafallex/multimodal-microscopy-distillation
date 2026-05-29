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
    """CAFNet-style: each modality's tokens attend to the OTHER modality (cross-attn)."""
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
        mbf, mfl = self.proj_bf(mbf), self.proj_fl(mfl)        # [B,dim,H,W]
        tbf = mbf.flatten(2).transpose(1, 2)                  # [B,HW,dim]
        tfl = mfl.flatten(2).transpose(1, 2)
        obf, _ = self.a_bf(tbf, tfl, tfl)                     # BF queries FL
        ofl, _ = self.a_fl(tfl, tbf, tbf)                     # FL queries BF
        tbf = self.n_bf(tbf + obf)
        tfl = self.n_fl(tfl + ofl)
        return tbf.mean(1), tfl.mean(1)                       # [B,dim] each


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
print("forward map spatial check:")
with torch.no_grad():
    fm = m.bf_feat(x_bf)
    print(f"  ResNet-50 feature map for 128x128 input: {tuple(fm.shape)} -> {fm.shape[-1]*fm.shape[-2]} attention tokens")
