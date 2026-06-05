"""DECISIVE TEST: can we recover patients by clustering on stain/illumination
signatures (patient-specific, malignancy-agnostic)? If yes on train (12 known
patients), the same works on test -> patient-consensus aggregation -> big AUC lift.
"""
import os, re, warnings
import numpy as np, pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.abspath(os.path.join(HERE, "..", "..", "multimodal-cancer-classification-challenge-2026"))
PAT = re.compile(r"^pat_(\d+)_image_\d+\.jpg$")
POS = {3, 5, 16, 17, 18}

def sig_feats(name):
    """Patient-signature features: global stain/illumination, NOT cell morphology."""
    f = {}
    for mod in ("BF", "FL"):
        im = Image.open(os.path.join(D, mod, "train", name))
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        g = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        for ci, ch in enumerate("rgb"):
            f[f"{mod}_{ch}_mean"] = float(rgb[:, :, ci].mean())
            f[f"{mod}_{ch}_std"] = float(rgb[:, :, ci].std())
        for q in (5, 50, 95):
            f[f"{mod}_p{q}"] = float(np.percentile(g, q))
        f[f"{mod}_mean"] = float(g.mean())
    return f

tr = pd.read_csv(D + "/train.csv"); tr["pid"] = tr["Name"].map(lambda n: int(PAT.match(n).group(1)))
parts = [d.sample(n=min(1000, len(d)), random_state=0) for _, d in tr.groupby("pid")]
samp = pd.concat(parts).reset_index(drop=True)
print(f"clustering {len(samp)} cells from {samp.pid.nunique()} true patients")

F = pd.DataFrame(Parallel(n_jobs=-1)(delayed(sig_feats)(n) for n in samp["Name"]))
Xs = StandardScaler().fit_transform(F.values)
true = samp["pid"].to_numpy()

for K in (12, 14, 20):
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(Xs)
    ag = AgglomerativeClustering(n_clusters=K).fit_predict(Xs)
    for tag, lab in (("KMeans", km), ("Agglom", ag)):
        ari = adjusted_rand_score(true, lab); nmi = normalized_mutual_info_score(true, lab)
        # cross-class contamination: fraction of clusters that mix pos+neg patients
        bad = 0
        for c in np.unique(lab):
            pats_in = set(true[lab == c])
            if (pats_in & POS) and (pats_in - POS):
                bad += 1
        # purity: each cluster's dominant-patient share
        pur = np.mean([np.bincount(true[lab == c]).max() / (lab == c).sum() for c in np.unique(lab)])
        print(f"  K={K:2d} {tag}: ARI={ari:.3f} NMI={nmi:.3f} purity={pur:.3f} "
              f"cross-class clusters={bad}/{K}")
