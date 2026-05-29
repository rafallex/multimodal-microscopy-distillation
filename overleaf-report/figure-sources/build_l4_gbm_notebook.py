"""Builder for the L4 feature-GBM Kaggle notebook (CPU, orthogonal ensemble member).
Emits notebooks/l4_feature_gbm_source.ipynb -- self-contained, nbformat-valid.
NOT run on Kaggle; this just assembles the notebook.
"""
import json, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "l4_feature_gbm_source.ipynb"

md0 = r"""# Multimodal Cancer Challenge 2026 — L4: Morphological/Texture Feature GBM (CPU, orthogonal)

A **decorrelated, GPU-free** ensemble member. Instead of a CNN, this extracts ~87
hand-crafted features per cell — intensity stats, morphology (Otsu region props),
GLCM texture, LBP, and **cross-modal BF/FL correlation** — from the 128×128 BF+FL crops,
then trains a LightGBM classifier.

**Why this exists.** Labels are *patient-level* (only 12 training patients), so every
strong CNN you have is correlated (rank-corr > 0.94) and blending them is capped. A feature
space orthogonal to the CNNs adds genuinely new signal. Local **leave-patients-out** CV:
~0.83 AUC (FL-only 0.76, BF-only 0.72 → complementary). Runs in ~20 min on Kaggle's 4 CPU
cores — **costs zero GPU quota**.

**How to use**
1. Attach the **`a3-adl`** dataset (required) and, optionally, **`submissionv47seed2`**
   (enables a ready-made rank-averaged blend + a decorrelation report).
2. Accelerator can stay **None/CPU** (no GPU needed). `Run All`.
3. Submit **`submission_gbm.csv`**. Then rank-average it with your CNN fleet in the
   cross-arch ensemble notebook — that blend is where the lift comes from.
"""

c_imports = r'''# === imports (all pre-installed on Kaggle) ===
import os, re, io, time, warnings
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import skew, kurtosis, rankdata
from joblib import Parallel, delayed
from skimage.filters import threshold_otsu, sobel
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
print("lightgbm", lgb.__version__)
'''

c_feats = r'''# === per-cell feature extractor (BF + FL grayscale, 128x128) ===
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
        for k in ("area","ecc","solidity","extent","eqdiam","perim","axisratio","ncomp"):
            f[f"{p}_{k}"] = 0.0
    return f

def texture_feats(g, p):
    f = {}
    q = (g * 31).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 2], angles=[0, np.pi/2], levels=32,
                        symmetric=True, normed=True)
    for prop in ("contrast","dissimilarity","homogeneity","energy","correlation","ASM"):
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
    f["fl_bf_mean_diff"]  = float(fl.mean() - bf.mean())
    f["fl_bf_corr"] = (float(np.corrcoef(bf.ravel(), fl.ravel())[0, 1])
                       if bf.std() > 1e-6 and fl.std() > 1e-6 else 0.0)
    return f

def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
print("feature functions ready")
'''

c_config = r'''# === config + paths (same candidate logic as the CNN fleet) ===
from pathlib import Path
DATA_ROOT_CANDIDATES = [
    Path("/kaggle/input/datasets/rafaelproena/a3-adl"),
    Path("/kaggle/input/a3-adl"),
    Path("/kaggle/input/competitions/multimodal-cancer-classification-challenge-2026"),
]
DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if (p / "train.csv").exists()), None)
assert DATA_ROOT is not None, f"train.csv not found at any of {DATA_ROOT_CANDIDATES}"
print("DATA_ROOT =", DATA_ROOT)

# optional teacher (for the rank-avg blend + decorrelation report)
_TEACHER_CANDIDATES = [
    "/kaggle/input/datasets/rafaelproena/submissionv47seed2/submission_seed2.csv",
    "/kaggle/input/submissionv47seed2/submission_seed2.csv",
]
TEACHER_CSV = next((p for p in _TEACHER_CANDIDATES if Path(p).exists()), None)
print("TEACHER_CSV =", TEACHER_CSV if TEACHER_CSV else "(none attached — blend step will skip)")

WORK = Path("/kaggle/working")
N_PER_PAT   = 8000     # cap train cells per patient (speed); None = use all
GBM_SEEDS   = [1, 2, 3, 4]
N_JOBS      = -1       # use all CPU cores
PAT_RE = re.compile(r"^pat_(\d+)_image_\d+\.jpg$")
POS_PATS = [3, 5, 16, 17, 18]
NEG_PATS = [7, 9, 10, 11, 13, 14, 15]
LGB_PARAMS = dict(objective="binary", metric="auc", num_leaves=31, learning_rate=0.04,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                  min_data_in_leaf=100, verbose=-1)
print("config ready  | GBM_SEEDS", GBM_SEEDS, "| N_PER_PAT", N_PER_PAT)
'''

c_extract = r'''# === extract features for train (capped) + test (all) ===
def parse_pid(name):
    m = PAT_RE.match(name)
    return int(m.group(1)) if m else -1

df_train = pd.read_csv(DATA_ROOT / "train.csv"); df_train.columns = [c.strip() for c in df_train.columns]
df_train["pid"] = df_train["Name"].map(parse_pid)
df_test = pd.read_csv(DATA_ROOT / "sampleSubmission.csv"); df_test.columns = [c.strip() for c in df_test.columns]
print(f"train {len(df_train)} cells / {df_train.pid.nunique()} patients | test {len(df_test)} cells")

if N_PER_PAT:
    parts = [d.sample(n=min(N_PER_PAT, len(d)), random_state=0) for _, d in df_train.groupby("pid")]
    df_train_s = pd.concat(parts).reset_index(drop=True)
else:
    df_train_s = df_train
print(f"using {len(df_train_s)} train cells for feature extraction")

def feat_for(name, split):
    bf = load_gray(DATA_ROOT / "BF" / split / name)
    fl = load_gray(DATA_ROOT / "FL" / split / name)
    return cell_features(bf, fl)

CACHE_TR, CACHE_TE = WORK / "Xtr.parquet", WORK / "Xte.parquet"
if CACHE_TR.exists() and CACHE_TE.exists():
    Xtr = pd.read_parquet(CACHE_TR); Xte = pd.read_parquet(CACHE_TE)
    print("loaded cached features")
else:
    t0 = time.time()
    ftr = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(feat_for)(n, "train") for n in df_train_s["Name"])
    print(f"train features in {time.time()-t0:.0f}s"); t0 = time.time()
    fte = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(feat_for)(n, "test") for n in df_test["Name"])
    print(f"test features in {time.time()-t0:.0f}s")
    Xtr = pd.DataFrame(ftr); Xtr.insert(0, "Name", df_train_s["Name"].values)
    Xtr["pid"] = df_train_s["pid"].values; Xtr["y"] = df_train_s["Diagnosis"].values
    Xte = pd.DataFrame(fte); Xte.insert(0, "Name", df_test["Name"].values)
    try:
        Xtr.to_parquet(CACHE_TR); Xte.to_parquet(CACHE_TE); print("cached features")
    except Exception as e:
        print("cache skipped:", e)

FEAT_COLS = [c for c in Xtr.columns if c not in ("Name", "pid", "y")]
print(f"n_features = {len(FEAT_COLS)}")
'''

c_cv = r'''# === honest leave-(1 pos + 1 neg)-patient-out CV (the diagnostic) ===
xmod = ["fl_bf_mean_ratio", "fl_bf_mean_diff", "fl_bf_corr"]
bf_cols = [c for c in FEAT_COLS if c.startswith("bf_")]
fl_cols = [c for c in FEAT_COLS if c.startswith("fl_") and c not in xmod]
colsets = {"BOTH": FEAT_COLS, "BF-only": bf_cols, "FL-only": fl_cols + xmod}

y_all = Xtr["y"].to_numpy(); pid_all = Xtr["pid"].to_numpy()
rng = np.random.default_rng(0)
folds = [(int(rng.choice(POS_PATS)), int(rng.choice(NEG_PATS))) for _ in range(6)]
print("CV folds (val pos,neg):", folds, "\n")
cv = {k: [] for k in colsets}
for vp, vn in folds:
    va = np.isin(pid_all, [vp, vn]); trm = ~va
    for tag, cols in colsets.items():
        m = lgb.train(LGB_PARAMS, lgb.Dataset(Xtr.loc[trm, cols], label=y_all[trm]), num_boost_round=300)
        cv[tag].append(roc_auc_score(y_all[va], m.predict(Xtr.loc[va, cols])))
for tag, a in cv.items():
    a = np.array(a)
    print(f"  {tag:9s}: grouped-val AUC = {a.mean():.4f} +/- {a.std():.4f}   {np.round(a,3)}")
'''

c_train = r'''# === final GBM: train multiple seeds on ALL sampled train cells, predict test ===
dtrain_all = lgb.Dataset(Xtr[FEAT_COLS], label=Xtr["y"].to_numpy())
test_preds = np.zeros(len(Xte))
for s in GBM_SEEDS:
    p = dict(LGB_PARAMS); p["seed"] = s; p["bagging_seed"] = s; p["feature_fraction_seed"] = s
    m = lgb.train(p, dtrain_all, num_boost_round=400)
    test_preds += m.predict(Xte[FEAT_COLS]) / len(GBM_SEEDS)

sub = pd.DataFrame({"Name": Xte["Name"].values, "Diagnosis": test_preds})
sub.to_csv(WORK / "submission_gbm.csv", index=False)
print(f"wrote submission_gbm.csv  | mean={test_preds.mean():.4f} "
      f"min={test_preds.min():.4f} max={test_preds.max():.4f}")
'''

c_blend = r'''# === optional: decorrelation report + ready-made rank-avg blend with the CNN teacher ===
if TEACHER_CSV:
    t = pd.read_csv(TEACHER_CSV).set_index("Name")["Diagnosis"]
    g = sub.set_index("Name")["Diagnosis"].reindex(t.index)
    sp = pd.concat([g, t], axis=1, keys=["gbm", "cnn"]).corr(method="spearman").iloc[0, 1]
    print(f"Spearman(GBM, CNN teacher) = {sp:.4f}   "
          f"(lower = more orthogonal = more ensemble lift)")
    gr = rankdata(g.values) / len(g); tr_ = rankdata(t.values) / len(t)
    blend = pd.DataFrame({"Name": t.index, "Diagnosis": 0.5*gr + 0.5*tr_})
    blend.to_csv(WORK / "submission_gbm_rankavg_v47s2.csv", index=False)
    print("wrote submission_gbm_rankavg_v47s2.csv  (50/50 rank-avg GBM + v47_seed2)")
    print("TIP: submit BOTH submission_gbm.csv (pure orthogonal) and this blend; the blend")
    print("     usually scores higher. For the full fleet, rank-average GBM in the cross-arch nb.")
else:
    print("No teacher attached -> skipping blend. Attach 'submissionv47seed2' to enable it.")
'''

c_imp = r'''# === feature importance (which signals the GBM relies on) ===
m = lgb.train(LGB_PARAMS, dtrain_all, num_boost_round=400)
imp = pd.Series(m.feature_importance("gain"), index=FEAT_COLS).sort_values(ascending=False)
print("top 20 features by gain:"); print(imp.head(20).round(0).to_string())
print("\nFiles in /kaggle/working:")
for f in sorted(WORK.glob("submission*.csv")):
    print("  ", f.name)
'''

cells = [("markdown", md0), ("code", c_imports), ("code", c_feats), ("code", c_config),
         ("code", c_extract), ("code", c_cv), ("code", c_train), ("code", c_blend), ("code", c_imp)]

def mkcell(kind, src):
    lines = src.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines, "id": uuid.uuid4().hex[:12]}
    if kind == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base

nb = {
    "cells": [mkcell(k, s) for k, s in cells],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("WROTE", OUT)
# validate
reloaded = json.loads(OUT.read_text(encoding="utf-8"))
print("cells:", len(reloaded["cells"]), "| nbformat:", reloaded["nbformat"], ".", reloaded["nbformat_minor"])
