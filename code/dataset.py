"""
Dataset for the Multimodal Cancer Classification Challenge 2026.

Pairs brightfield (BF) and fluorescence (FL) images of the same cell.
Train filenames look like:  pat_NNN_image_K.jpg
Test  filenames look like:  image_K.jpg
The same image filename exists in both BF/ and FL/ subfolders.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Regex for the patient id, e.g. "pat_07_image_1234.jpg" -> 7
_PAT_RE = re.compile(r"^pat_(\d+)_image_\d+\.jpg$")


def parse_patient_id(filename: str) -> Optional[int]:
    """Return the patient id from a train filename, or None for test filenames."""
    m = _PAT_RE.match(Path(filename).name)
    return int(m.group(1)) if m else None


class CellDataset(Dataset):
    """Returns dict(bf, fl, label, name).

    Parameters
    ----------
    df             : DataFrame with column "Name" (and "Diagnosis" for train).
    bf_dir, fl_dir : Directories holding the BF and FL JPEGs.
    bf_transform   : Callable(PIL.Image) -> Tensor applied to the BF image.
    fl_transform   : Callable(PIL.Image) -> Tensor applied to the FL image.
    paired_transform : Optional callable(bf_tensor, fl_tensor) -> (bf, fl) for
                       augmentations that must stay aligned across modalities.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        bf_dir: Path,
        fl_dir: Path,
        bf_transform: Callable,
        fl_transform: Callable,
        paired_transform: Optional[Callable] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.bf_dir = Path(bf_dir)
        self.fl_dir = Path(fl_dir)
        self.bf_transform = bf_transform
        self.fl_transform = fl_transform
        self.paired_transform = paired_transform

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load(path: Path) -> Image.Image:
        # Open as grayscale - both modalities are single-channel microscopy images.
        return Image.open(path).convert("L")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        name = row["Name"]
        bf = self._load(self.bf_dir / name)
        fl = self._load(self.fl_dir / name)
        bf = self.bf_transform(bf)
        fl = self.fl_transform(fl)
        if self.paired_transform is not None:
            bf, fl = self.paired_transform(bf, fl)
        label = int(row["Diagnosis"]) if "Diagnosis" in row else -1
        return {"bf": bf, "fl": fl, "label": label, "name": name}


def load_train_df(train_csv: Path) -> pd.DataFrame:
    """Load train.csv and add a patient_id column."""
    df = pd.read_csv(train_csv)
    # Some hosts add a leading space after the comma; normalize column names.
    df.columns = [c.strip() for c in df.columns]
    df["patient_id"] = df["Name"].map(parse_patient_id)
    if df["patient_id"].isna().any():
        bad = df[df["patient_id"].isna()].head()
        raise ValueError(f"Some train rows have no parseable patient_id, e.g.:\n{bad}")
    df["patient_id"] = df["patient_id"].astype(int)
    return df


def load_test_df(sample_submission_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(sample_submission_csv)
    df.columns = [c.strip() for c in df.columns]
    return df
