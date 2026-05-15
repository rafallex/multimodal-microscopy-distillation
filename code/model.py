"""
Multimodal classifier.

Two backbones (one for BF, one for FL), each takes a 1-channel input,
features concatenated and fed to a small head.

Supported backbones (set via backbone= param):
  'resnet18'        — lightweight, fast, good baseline.
  'efficientnet_b0' — stronger (~3× params), requires timm.

Why two separate backbones (vs. stacking BF+FL as 2-channel input):
- BF and FL come from different physics; their low-level statistics differ.
- Separate branches let us later freeze/pretrain each independently.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


# ---------------------------------------------------------------------------
# Branch factories
# ---------------------------------------------------------------------------

def _make_resnet18_branch(pretrained: bool = True):
    """ResNet-18, 1-channel input, no final FC. Returns (module, feat_dim)."""
    weights = "DEFAULT" if pretrained else None
    net = models.resnet18(weights=weights)
    w = net.conv1.weight.data  # [64, 3, 7, 7]
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        new_conv.weight.data = w.mean(dim=1, keepdim=True)
    net.conv1 = new_conv
    fd = net.fc.in_features
    net.fc = nn.Identity()
    return net, fd


def _make_effnet_b0_branch(pretrained: bool = True):
    """EfficientNet-B0 (timm), 1-channel input, no classifier. Returns (module, feat_dim)."""
    if not _HAS_TIMM:
        raise ImportError("timm is required for efficientnet_b0: pip install timm")
    net = timm.create_model("efficientnet_b0", pretrained=pretrained,
                             num_classes=0, global_pool="avg")
    old = net.conv_stem
    new_conv = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                         stride=old.stride, padding=old.padding, bias=False)
    if pretrained:
        new_conv.weight.data = old.weight.data.mean(dim=1, keepdim=True)
    net.conv_stem = new_conv
    fd = net.num_features  # 1280
    return net, fd


def _make_branch(backbone: str = "resnet18", pretrained: bool = True):
    """Factory dispatching to the per-backbone constructor.

    Parameters
    ----------
    backbone : {"resnet18", "efficientnet_b0"}
        Which CNN family to use for the branch.
    pretrained : bool
        If True, load ImageNet weights and average the 3-channel input conv
        across channels (since our inputs are 1-channel grayscale microscopy).

    Returns
    -------
    (nn.Module, int)
        The branch network (final classifier head replaced with Identity)
        and the resulting feature dimensionality.
    """
    if backbone == "efficientnet_b0":
        return _make_effnet_b0_branch(pretrained)
    return _make_resnet18_branch(pretrained)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class MultimodalClassifier(nn.Module):
    """Two-branch multimodal classifier with late fusion.

    Separate CNN branches encode BF and FL inputs (each 1-channel HxW), the
    resulting feature vectors are concatenated and passed through a small MLP
    head (Linear → BN → ReLU → Dropout → Linear) that outputs a single logit
    per cell. Train with `BCEWithLogitsLoss`.

    Parameters
    ----------
    pretrained : bool
        Whether to initialise the branches from ImageNet weights.
    dropout : float
        Dropout rate in the fusion head (applied after ReLU).
    backbone : str
        Branch architecture; see `_make_branch`.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3,
                 backbone: str = "resnet18"):
        super().__init__()
        self.bf_branch, fd = _make_branch(backbone, pretrained)
        self.fl_branch, _  = _make_branch(backbone, pretrained)
        hidden = 512 if fd >= 512 else 256
        self.head = nn.Sequential(
            nn.Linear(fd * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, bf: torch.Tensor, fl: torch.Tensor) -> torch.Tensor:
        """Return a logit (not probability) of shape ``[B]`` for the batch."""
        feat = torch.cat([self.bf_branch(bf), self.fl_branch(fl)], dim=1)
        return self.head(feat).squeeze(-1)  # [B]


class SingleModalClassifier(nn.Module):
    """Single-branch classifier — for ablations isolating BF or FL alone.

    Useful for answering "how much does the second modality help?". Same
    backbone family as `MultimodalClassifier` but with one branch.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3,
                 backbone: str = "resnet18"):
        super().__init__()
        self.branch, fd = _make_branch(backbone, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fd, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a logit of shape ``[B]`` for a single-modality batch."""
        return self.head(self.branch(x)).squeeze(-1)
