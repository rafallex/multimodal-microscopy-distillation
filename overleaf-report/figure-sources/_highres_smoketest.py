"""CPU smoke-test for v60's resolution change: confirm (a) the train transform
upscales a real 128px crop to 192px, and (b) the dual EffNet-B0 model accepts 192px
input and learns. EffNet is fully-conv + global-pool, so size-agnostic -- this just
verifies the plumbing.
"""
import os, torch, torch.nn as nn, timm
import torchvision.transforms as T
from PIL import Image
import numpy as np

TRAIN_RES = 192
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.abspath(os.path.join(HERE, "..", "..", "multimodal-cancer-classification-challenge-2026"))

# (a) transform: Resize(192) prepended, on a real 128 image
img = Image.open(os.path.join(D, "BF", "train", os.listdir(os.path.join(D, "BF", "train"))[0])).convert("L")
tf = T.Compose([T.Resize((TRAIN_RES, TRAIN_RES)), T.ColorJitter(0.4, 0.4),
                lambda im: T.functional.normalize(T.functional.to_tensor(im), [0.5], [0.5])])
t = tf(img)
print(f"transform OK: input {img.size} PIL -> tensor {tuple(t.shape)} (expect (1,192,192))")
assert t.shape == (1, 192, 192)


def _branch(pretrained=False):
    net = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0, global_pool="avg")
    old = net.conv_stem
    c = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    net.conv_stem = c
    return net, net.num_features


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.bf, fd = _branch(); self.fl, _ = _branch()
        self.head = nn.Sequential(nn.Linear(fd*2, 512), nn.BatchNorm1d(512), nn.ReLU(True),
                                  nn.Dropout(0.4), nn.Linear(512, 1))
    def forward(self, bf, fl):
        return self.head(torch.cat([self.bf(bf), self.fl(fl)], 1)).squeeze(-1)


torch.manual_seed(0)
m = M()
xb, xf = torch.randn(4, 1, 192, 192), torch.randn(4, 1, 192, 192)
out = m(xb, xf)
print(f"model OK at 192px: out {tuple(out.shape)} (expect (4,))   params {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
assert out.shape == (4,)
y = torch.tensor([1., 0., 1., 0.]); opt = torch.optim.AdamW(m.parameters(), 1e-3); lf = nn.BCEWithLogitsLoss()
ls = []
for _ in range(15):
    opt.zero_grad(); l = lf(m(xb, xf), y); l.backward(); opt.step(); ls.append(l.item())
print(f"learns at 192px: loss {ls[0]:.4f} -> {ls[-1]:.4f}  ({'OK' if ls[-1] < ls[0]*0.6 else 'flat?'})")
print("ALL OK -- v60 resolution plumbing sound")
