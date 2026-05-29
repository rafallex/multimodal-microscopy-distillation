"""Build the L1 ensemble-teacher CSV for v56 distillation.

A distillation teacher should be a *probability* mean (preserves soft-confidence
"dark knowledge"), NOT a rank-average (which flattens the scale). We average the
three v47 seeds -> a smoother, lower-variance teacher than the lucky seed-2.

Upload results/teacher_ensemble/teacher_v47seeds_mean.csv to Kaggle as a dataset
named `teacherv47ensemble` so v56's path candidates resolve it.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = {
    "v47_s1": ROOT / "results/v47/submission_seed1.csv",
    "v47_s2": ROOT / "results/v47/submission_seed2.csv",
    "v47_s3": ROOT / "results/v47/submission_seed3.csv",
    "v46_ens": ROOT / "results/v46/submission.csv",
}
M = pd.DataFrame({k: pd.read_csv(p).set_index("Name")["Diagnosis"] for k, p in SRC.items()})
assert not M.isna().any().any(), "Name misalignment across teacher CSVs"

out_dir = ROOT / "results/teacher_ensemble"
out_dir.mkdir(exist_ok=True)
for name, cols in (("teacher_v47seeds_mean.csv", ["v47_s1", "v47_s2", "v47_s3"]),
                   ("teacher_v47seeds_plus_v46_mean.csv", list(SRC))):
    t = M[cols].mean(axis=1)
    pd.DataFrame({"Name": t.index, "Diagnosis": t.values}).to_csv(out_dir / name, index=False)
    print(f"wrote {name:38s} mean={t.mean():.4f} std={t.std():.4f}")
print("Spearman within seeds:\n", M.corr(method="spearman").round(3).to_string())
