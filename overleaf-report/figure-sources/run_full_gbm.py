"""Full L4 GBM pipeline, run LOCALLY on CPU. Produces real, uploadable CSVs:
  results/l4_gbm/submission_gbm.csv                  (pure orthogonal GBM)
  results/l4_gbm/submission_blend_gbm50_v47s2.csv    (50/50 rank-avg with v47_seed2)
  results/l4_gbm/submission_blend_gbm35_v47s2.csv    (35/65, favouring the stronger CNN)
Caches features to _l4_cache/ so re-runs (richer models) are instant.
"""
import os, re, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from _l4_features import cell_features, load_gray
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
D = os.path.join(ROOT, "multimodal-cancer-classification-challenge-2026")
CACHE = os.path.join(ROOT, "_l4_cache"); os.makedirs(CACHE, exist_ok=True)
OUTD = os.path.join(ROOT, "results", "l4_gbm"); os.makedirs(OUTD, exist_ok=True)
PAT = re.compile(r"^pat_(\d+)_image_\d+\.jpg$")
POS, NEG = [3, 5, 16, 17, 18], [7, 9, 10, 11, 13, 14, 15]

def feat_for(name, split):
    bf = load_gray(os.path.join(D, "BF", split, name))
    fl = load_gray(os.path.join(D, "FL", split, name))
    return cell_features(bf, fl)

def extract(names, split, tag):
    cf = os.path.join(CACHE, f"X_{tag}.pkl")
    if os.path.exists(cf):
        print(f"[{tag}] cached"); return pd.read_pickle(cf)
    t0 = time.time()
    rows = Parallel(n_jobs=-1, verbose=2)(delayed(feat_for)(n, split) for n in names)
    X = pd.DataFrame(rows); X.insert(0, "Name", list(names))
    X.to_pickle(cf)
    print(f"[{tag}] {len(names)} cells in {time.time()-t0:.0f}s -> cached")
    return X

tr = pd.read_csv(D + "/train.csv"); tr["pid"] = tr["Name"].map(lambda n: int(PAT.match(n).group(1)))
te = pd.read_csv(D + "/sampleSubmission.csv")
print(f"train {len(tr)} | test {len(te)}")

Xtr = extract(tr["Name"].tolist(), "train", "train")
Xtr["pid"] = tr["pid"].values; Xtr["y"] = tr["Diagnosis"].values
Xte = extract(te["Name"].tolist(), "test", "test")
FEAT = [c for c in Xtr.columns if c not in ("Name", "pid", "y")]
print(f"n_features={len(FEAT)}")

PARAMS = dict(objective="binary", metric="auc", num_leaves=31, learning_rate=0.03,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              min_data_in_leaf=200, verbose=-1)

# grouped CV (8 folds) for an honest estimate
y, pid = Xtr["y"].to_numpy(), Xtr["pid"].to_numpy()
rng = np.random.default_rng(0)
folds = [(int(rng.choice(POS)), int(rng.choice(NEG))) for _ in range(8)]
aucs = []
for vp, vn in folds:
    va = np.isin(pid, [vp, vn]); trm = ~va
    m = lgb.train(PARAMS, lgb.Dataset(Xtr.loc[trm, FEAT], label=y[trm]), num_boost_round=400)
    aucs.append(roc_auc_score(y[va], m.predict(Xtr.loc[va, FEAT])))
aucs = np.array(aucs)
print(f"grouped-CV AUC = {aucs.mean():.4f} +/- {aucs.std():.4f}  {np.round(aucs,3)}")

# final: 4 seeds on all train -> predict test
dall = lgb.Dataset(Xtr[FEAT], label=y)
preds = np.zeros(len(Xte))
for s in (1, 2, 3, 4):
    p = dict(PARAMS); p["seed"] = s; p["bagging_seed"] = s; p["feature_fraction_seed"] = s
    preds += lgb.train(p, dall, num_boost_round=500).predict(Xte[FEAT]) / 4
pd.DataFrame({"Name": Xte["Name"], "Diagnosis": preds}).to_csv(os.path.join(OUTD, "submission_gbm.csv"), index=False)
print(f"wrote submission_gbm.csv mean={preds.mean():.4f}")

# blends with v47_seed2
t = pd.read_csv(os.path.join(ROOT, "results", "v47", "submission_seed2.csv")).set_index("Name")["Diagnosis"]
g = pd.Series(preds, index=Xte["Name"].values).reindex(t.index)
sp = pd.concat([g, t], axis=1, keys=["g", "t"]).corr(method="spearman").iloc[0, 1]
print(f"Spearman(GBM, v47_seed2) = {sp:.4f}")
gr, tr_ = rankdata(g.values) / len(g), rankdata(t.values) / len(t)
for wg, name in ((0.50, "submission_blend_gbm50_v47s2.csv"), (0.35, "submission_blend_gbm35_v47s2.csv")):
    pd.DataFrame({"Name": t.index, "Diagnosis": wg*gr + (1-wg)*tr_}).to_csv(os.path.join(OUTD, name), index=False)
    print("wrote", name, f"(GBM weight {wg})")
print("DONE")
