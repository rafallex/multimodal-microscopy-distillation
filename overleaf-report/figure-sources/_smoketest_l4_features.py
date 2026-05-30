"""Smoke-test the L4 morphological/texture feature extractor on real local images.
Measures per-cell timing and a patient-grouped AUC (both/BF-only/FL-only) to confirm
the features carry orthogonal signal before committing the full Kaggle notebook.
NOT run on Kaggle -- local validation only.
"""
import os, time, re, io
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import skew, kurtosis
from skimage.filters import threshold_otsu, sobel
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

D = "multimodal-cancer-classification-challenge-2026"
PAT_RE = re.compile(r"^pat_(\d+)_image_\d+\.jpg$")


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


# ---- sample ~1800 cells across 6 patients, grouped train/val ----
tr = pd.read_csv(os.path.join(D, "train.csv"))
tr["pid"] = tr["Name"].map(lambda n: int(PAT_RE.match(n).group(1)))
pats = sorted(tr["pid"].unique())
print("patients:", pats)
sample_pats = pats[:6]
rows = []
for pid in sample_pats:
    sub = tr[tr.pid == pid].sample(n=min(300, (tr.pid == pid).sum()), random_state=0)
    rows.append(sub)
samp = pd.concat(rows).reset_index(drop=True)
print("sample cells:", len(samp), "from patients", sample_pats)

t0 = time.time()
X, y, grp = [], [], []
for _, r in samp.iterrows():
    bf = load_gray(os.path.join(D, "BF", "train", r["Name"]))
    fl = load_gray(os.path.join(D, "FL", "train", r["Name"]))
    X.append(cell_features(bf, fl)); y.append(int(r["Diagnosis"])); grp.append(int(r["pid"]))
dt = time.time() - t0
Xdf = pd.DataFrame(X)
per_cell_ms = 1000 * dt / len(samp)
print(f"\nextracted {len(samp)} cells in {dt:.1f}s  ->  {per_cell_ms:.1f} ms/cell")
print(f"projected for 173,342 cells (single thread): {per_cell_ms*173342/1000/60:.1f} min")
print(f"  with 4 Kaggle CPU cores (joblib): ~{per_cell_ms*173342/1000/60/4:.1f} min")
print(f"n_features = {Xdf.shape[1]}")

# ---- patient-grouped signal check: train on 4 patients, val on 2 ----
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
g = np.array(grp); y = np.array(y)
val_pats = sample_pats[-2:]
tr_mask = ~np.isin(g, val_pats); va_mask = np.isin(g, val_pats)
bf_cols = [c for c in Xdf.columns if c.startswith("bf")]
fl_cols = [c for c in Xdf.columns if c.startswith("fl")]
print(f"\nval patients = {val_pats}  (train {tr_mask.sum()} / val {va_mask.sum()} cells)")
for tag, cols in (("BOTH", list(Xdf.columns)), ("BF-only", bf_cols + ["fl_bf_mean_ratio"]),
                  ("FL-only", fl_cols)):
    dtr = lgb.Dataset(Xdf.loc[tr_mask, cols], label=y[tr_mask])
    params = dict(objective="binary", metric="auc", num_leaves=31, learning_rate=0.05,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
    m = lgb.train(params, dtr, num_boost_round=200)
    pv = m.predict(Xdf.loc[va_mask, cols])
    try:
        auc = roc_auc_score(y[va_mask], pv)
    except Exception:
        auc = float("nan")
    print(f"  {tag:9s} ({len(cols):2d} feats): grouped-val AUC = {auc:.4f}")

# top features
dtr = lgb.Dataset(Xdf.loc[tr_mask], label=y[tr_mask])
m = lgb.train(dict(objective="binary", metric="auc", num_leaves=31, learning_rate=0.05, verbose=-1),
              dtr, num_boost_round=200)
imp = pd.Series(m.feature_importance(importance_type="gain"), index=Xdf.columns).sort_values(ascending=False)
print("\ntop 15 features by gain:")
print(imp.head(15).round(1).to_string())
