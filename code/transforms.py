"""
Augmentations.

Important: BF and FL of the same cell must receive the *same* geometric
augmentation (flip, rotate, crop) so the two views stay aligned. Photometric
augmentations (brightness, contrast, noise) can differ.

Changes from baseline:
- PairedGeoAug now samples a random 90° step (0/90/180/270°) before the fine
  rotation. Cells have no canonical orientation, so full rotational invariance
  is important.
- Stronger ColorJitter (0.3) and fine rotation reduced to ±15° (on top of 90°).
- RandomErasing added to training transforms to regularize texture overfitting.
"""
from __future__ import annotations

import random
from typing import Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# Per-modality normalization. Computed from 500 random train images on
# 2026-05-12. BF is roughly mid-gray; FL is mostly dark with sparse bright
# fluorescent spots, so its mean is much lower.
BF_MEAN, BF_STD = 0.504, 0.216
FL_MEAN, FL_STD = 0.100, 0.144


def to_tensor_bf(img):
    """PIL grayscale image -> normalized 1xHxW tensor (BF statistics)."""
    t = TF.to_tensor(img)  # [1, H, W] in [0, 1]
    return TF.normalize(t, [BF_MEAN], [BF_STD])


def to_tensor_fl(img):
    """PIL grayscale image -> normalized 1xHxW tensor (FL statistics).

    FL has a much darker mean than BF because most pixels are unlit
    background with sparse bright fluorescent spots.
    """
    t = TF.to_tensor(img)
    return TF.normalize(t, [FL_MEAN], [FL_STD])


class PairedGeoAug:
    """Apply identical random flip / rotation to BF and FL tensors.

    Applies (in order):
      1. Random 90° step (k ∈ {0,1,2,3}) — captures full rotational symmetry.
      2. Random hflip / vflip.
      3. Fine uniform rotation in [-max_rot, max_rot] degrees.
    """

    def __init__(self, p_hflip: float = 0.5, p_vflip: float = 0.5,
                 rot90: bool = True, max_rot: float = 15.0):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.rot90 = rot90
        self.max_rot = max_rot

    def __call__(self, bf: torch.Tensor, fl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rot90:
            k = random.randint(0, 3)
            if k:
                bf = torch.rot90(bf, k, dims=(-2, -1))
                fl = torch.rot90(fl, k, dims=(-2, -1))
        if random.random() < self.p_hflip:
            bf, fl = TF.hflip(bf), TF.hflip(fl)
        if random.random() < self.p_vflip:
            bf, fl = TF.vflip(bf), TF.vflip(fl)
        if self.max_rot > 0:
            angle = random.uniform(-self.max_rot, self.max_rot)
            bf = TF.rotate(bf, angle)
            fl = TF.rotate(fl, angle)
        return bf, fl


def train_modality_transform(modality: str):
    """Photometric jitter + normalization + RandomErasing for training.

    Per-modality (BF or FL) because their grayscale statistics differ.
    Geometric augs (rotation, flips) are NOT included here -- those must be
    applied identically to BF and FL via PairedGeoAug to preserve cross-
    modal alignment.
    """
    norm = to_tensor_bf if modality == "bf" else to_tensor_fl
    return T.Compose([
        T.ColorJitter(brightness=0.3, contrast=0.3),
        norm,
        T.RandomErasing(p=0.2, scale=(0.02, 0.15), value=0),
    ])


def eval_modality_transform(modality: str):
    """Inference-time transform: normalize only, no augmentation."""
    return to_tensor_bf if modality == "bf" else to_tensor_fl
