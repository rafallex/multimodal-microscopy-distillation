"""End-to-end smoke test of the L4 notebook's train->predict->submit->blend logic
on small local data. Validates the parts the signal test didn't cover:
test-image extraction, submission writing, and the teacher rank-avg blend.
"""
import os, re, time, tempfile, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from _l4_features import cell_features, load_gray
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.abspath(os.path.join(HERE, "..", "..", "multimodal-cancer-classification-challenge-2026"))
TEACHER = os.path.abspath(os.path.join(HERE, "..", "..", "results", "v47", "submission_seed2.csv"))
WORK = tempfile.mkdtemp(prefix="l4e2e_")
PAT = re.compile(r"^pat_(\d+)_image_\d+\.jpg$")

tr = pd.read_csv(D + "/train.csv"); tr["pid"] = tr["Name"].map(lambda n: int(PAT.match(n).group(1)))
te = pd.read_csv(D + "/sampleSubmission.csv")
parts = [d.sample(n=min(200, len(d)), random_state=0) for _, d in tr.groupby("pid")]
trs = pd.concat(parts).reset_index(drop=True)
tes = te.sample(n=500, random_state=0).reset_index(drop=True)
print(f"train sample {len(trs)} | test sample {len(tes)}")

def feat_for(name, split):
    bf = load_gray(os.path.join(D, "BF", split, name))
    fl = load_gray(os.path.join(D, "FL", split, name))
    return cell_features(bf, fl)

t0 = time.time()
ftr = Parallel(n_jobs=-1)(delayed(feat_for)(n, "train") for n in trs["Name"])
fte = Parallel(n_jobs=-1)(delayed(feat_for)(n, "test") for n in tes["Name"])
print(f"extracted train+test in {time.time()-t0:.0f}s  (TEST EXTRACTION OK -> image_N.jpg pattern works)")
Xtr = pd.DataFrame(ftr); Xte = pd.DataFrame(fte)
FEAT = list(Xtr.columns)

# train K seeds on all train, predict test
dall = lgb.Dataset(Xtr, label=trs["Diagnosis"].to_numpy())
params = dict(objective="binary", metric="auc", num_leaves=31, learning_rate=0.04,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              min_data_in_leaf=100, verbose=-1)
preds = np.zeros(len(Xte))
for s in (1, 2, 3, 4):
    p = dict(params); p["seed"] = s
    m = lgb.train(p, dall, num_boost_round=200)
    preds += m.predict(Xte) / 4
sub = pd.DataFrame({"Name": tes["Name"].values, "Diagnosis": preds})
sub.to_csv(os.path.join(WORK, "submission_gbm.csv"), index=False)
print(f"wrote submission_gbm.csv  rows={len(sub)} cols={list(sub.columns)} "
      f"range=[{preds.min():.3f},{preds.max():.3f}] mean={preds.mean():.3f}")

# teacher blend (align on the 500 sampled test names)
t = pd.read_csv(TEACHER).set_index("Name")["Diagnosis"]
g = sub.set_index("Name")["Diagnosis"]
common = g.index.intersection(t.index)
gg, tt = g.loc[common], t.loc[common]
sp = pd.concat([gg, tt], axis=1, keys=["gbm", "cnn"]).corr(method="spearman").iloc[0, 1]
gr = rankdata(gg.values) / len(gg); tr_ = rankdata(tt.values) / len(tt)
blend = pd.DataFrame({"Name": common, "Diagnosis": 0.5*gr + 0.5*tr_})
blend.to_csv(os.path.join(WORK, "submission_gbm_rankavg_v47s2.csv"), index=False)
print(f"teacher blend OK: Spearman(GBM,CNN)={sp:.3f} on {len(common)} common names; "
      f"wrote blend rows={len(blend)}")
print("\nALL E2E STEPS PASSED. outputs in", WORK)
for f in sorted(os.listdir(WORK)):
    print("  ", f)
