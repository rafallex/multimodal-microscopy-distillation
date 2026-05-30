"""Honest leave-patients-out signal test for the L4 feature-GBM.
Samples cells from all 12 patients, extracts features in parallel, then does
leave-(1 pos + 1 neg)-patient-out CV. If grouped AUC >> 0.5 and stable, the
features transfer across patients and are worth ensembling with the CNN fleet.
"""
import os, re, time, itertools, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from _l4_features import cell_features, load_gray
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.abspath(os.path.join(_HERE, "..", "..", "multimodal-cancer-classification-challenge-2026"))
PAT = re.compile(r"^pat_(\d+)_image_\d+\.jpg$")
N_PER_PAT = 500

tr = pd.read_csv(D + "/train.csv")
tr["pid"] = tr["Name"].map(lambda n: int(PAT.match(n).group(1)))
parts = [d.sample(n=min(N_PER_PAT, len(d)), random_state=0) for _, d in tr.groupby("pid")]
samp = pd.concat(parts).reset_index(drop=True)
print(f"sampled {len(samp)} cells from {samp.pid.nunique()} patients")


def feat_row(name):
    bf = load_gray(os.path.join(D, "BF", "train", name))
    fl = load_gray(os.path.join(D, "FL", "train", name))
    return cell_features(bf, fl)


t0 = time.time()
feats = Parallel(n_jobs=4)(delayed(feat_row)(n) for n in samp["Name"])
print(f"extracted in {time.time()-t0:.1f}s")
X = pd.DataFrame(feats)
y = samp["Diagnosis"].to_numpy()
pid = samp["pid"].to_numpy()

pos_pats = [3, 5, 16, 17, 18]
neg_pats = [7, 9, 10, 11, 13, 14, 15]
xmod_cols = ["fl_bf_mean_ratio", "fl_bf_mean_diff", "fl_bf_corr"]
bf_cols = [c for c in X.columns if c.startswith("bf_")]
fl_cols = [c for c in X.columns if c.startswith("fl_") and c not in xmod_cols]
colsets = {"BOTH": list(X.columns), "BF-only": bf_cols, "FL-only": fl_cols + xmod_cols}

params = dict(objective="binary", metric="auc", num_leaves=31, learning_rate=0.05,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              min_data_in_leaf=50, verbose=-1)

# leave-(1 pos + 1 neg)-out: 6 representative folds
rng = np.random.default_rng(0)
pairs = [(rng.choice(pos_pats), rng.choice(neg_pats)) for _ in range(6)]
print(f"\nleave-(1pos+1neg)-out grouped CV, {len(pairs)} folds: {pairs}\n")
results = {k: [] for k in colsets}
for vp, vn in pairs:
    va = np.isin(pid, [vp, vn]); trm = ~va
    for tag, cols in colsets.items():
        m = lgb.train(params, lgb.Dataset(X.loc[trm, cols], label=y[trm]), num_boost_round=300)
        auc = roc_auc_score(y[va], m.predict(X.loc[va, cols]))
        results[tag].append(auc)
for tag, aucs in results.items():
    a = np.array(aucs)
    print(f"  {tag:9s}: grouped-val AUC = {a.mean():.4f} +/- {a.std():.4f}   (folds: {np.round(a,3)})")

# feature importance on a full-data fit
m = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=300)
imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print("\ntop 15 features by gain:")
print(imp.head(15).round(0).to_string())
