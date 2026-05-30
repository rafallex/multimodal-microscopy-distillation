"""L4 per-cell feature extractor (BF+FL grayscale, 128x128).
Importable module shared by the smoke/signal tests and the Kaggle notebook builder.
Pure numpy/scipy/skimage -- all pre-installed on Kaggle.
"""
import numpy as np
from PIL import Image
from scipy.stats import skew, kurtosis
from skimage.filters import threshold_otsu, sobel
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def _rp(props, *names, default=0.0):
    for n in names:
        if hasattr(props, n):
            try:
                return float(getattr(props, n))
            except Exception:
                pass
    return default


def intensity_feats(g, p):
    flat = g.ravel(); f = {}
    f[f"{p}_mean"], f[f"{p}_std"] = float(flat.mean()), float(flat.std())
    f[f"{p}_min"], f[f"{p}_max"] = float(flat.min()), float(flat.max())
    for q in (10, 25, 50, 75, 90):
        f[f"{p}_p{q}"] = float(np.percentile(flat, q))
    f[f"{p}_skew"], f[f"{p}_kurt"] = float(skew(flat)), float(kurtosis(flat))
    hist, _ = np.histogram(flat, bins=32, range=(0, 1), density=True)
    hist = hist / (hist.sum() + 1e-9)
    f[f"{p}_entropy"] = float(-(hist * np.log(hist + 1e-9)).sum())
    return f


def morph_feats(g, p):
    f = {}
    try:
        t = threshold_otsu(g)
    except Exception:
        t = 0.5
    mask = g > t
    f[f"{p}_fgfrac"] = float(mask.mean())
    f[f"{p}_fg_mean"] = float(g[mask].mean()) if mask.any() else 0.0
    f[f"{p}_bg_mean"] = float(g[~mask].mean()) if (~mask).any() else 0.0
    f[f"{p}_fg_bg_diff"] = f[f"{p}_fg_mean"] - f[f"{p}_bg_mean"]
    lbl = label(mask)
    if lbl.max() >= 1:
        pr = max(regionprops(lbl), key=lambda r: r.area)
        f[f"{p}_area"] = _rp(pr, "area") / g.size
        f[f"{p}_ecc"] = _rp(pr, "eccentricity")
        f[f"{p}_solidity"] = _rp(pr, "solidity")
        f[f"{p}_extent"] = _rp(pr, "extent")
        f[f"{p}_eqdiam"] = _rp(pr, "equivalent_diameter_area", "equivalent_diameter") / 128.0
        f[f"{p}_perim"] = _rp(pr, "perimeter") / 128.0
        major = _rp(pr, "axis_major_length", "major_axis_length") + 1e-6
        minor = _rp(pr, "axis_minor_length", "minor_axis_length")
        f[f"{p}_axisratio"] = minor / major
        f[f"{p}_ncomp"] = float(lbl.max())
    else:
        for k in ("area", "ecc", "solidity", "extent", "eqdiam", "perim", "axisratio", "ncomp"):
            f[f"{p}_{k}"] = 0.0
    return f


def texture_feats(g, p):
    f = {}
    q = (g * 31).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 2], angles=[0, np.pi / 2], levels=32,
                        symmetric=True, normed=True)
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        f[f"{p}_glcm_{prop}"] = float(graycoprops(glcm, prop).mean())
    lbp = local_binary_pattern((g * 255).astype(np.uint8), P=8, R=1, method="uniform")
    h, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    for i, v in enumerate(h):
        f[f"{p}_lbp{i}"] = float(v)
    sob = sobel(g)
    f[f"{p}_grad_mean"], f[f"{p}_grad_std"] = float(sob.mean()), float(sob.std())
    return f


def cell_features(bf, fl):
    """bf, fl: float32 [0,1] arrays. Returns dict of ~87 features."""
    f = {}
    for g, p in ((bf, "bf"), (fl, "fl")):
        f.update(intensity_feats(g, p)); f.update(morph_feats(g, p)); f.update(texture_feats(g, p))
    f["fl_bf_mean_ratio"] = (fl.mean() + 1e-6) / (bf.mean() + 1e-6)
    f["fl_bf_mean_diff"] = float(fl.mean() - bf.mean())
    f["fl_bf_corr"] = (float(np.corrcoef(bf.ravel(), fl.ravel())[0, 1])
                       if bf.std() > 1e-6 and fl.std() > 1e-6 else 0.0)
    return f


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
