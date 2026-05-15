# =============================================================================
# Kaggle notebook starter for Multimodal Cancer Classification Challenge 2026.
#
# Each block separated by "# %% [cell] ..." is meant to be a separate Kaggle
# notebook cell. Just paste them one by one (or use jupytext / vscode).
#
# Before running:
#   - Settings sidebar: Accelerator = GPU T4 x2 (or P100), Internet = On.
#   - Make sure the competition data is attached (it is by default if you
#     created the notebook from the comp page).
# =============================================================================

# %% [cell] 0: Environment check
import sys, subprocess, torch
print("python :", sys.version.split()[0])
print("torch  :", torch.__version__, "cuda:", torch.cuda.is_available(),
      "devices:", torch.cuda.device_count())
!nvidia-smi -L
DATA_ROOT = "/kaggle/input/multimodal-cancer-classification-challenge-2026"
!ls {DATA_ROOT}


# %% [cell] 1: Drop the code modules onto disk
# Simplest approach: paste the contents of dataset.py / splits.py / model.py /
# transforms.py / train.py / predict.py / profile_data.py into separate cells
# below, OR upload the `code/` folder as a Kaggle Dataset and copy it here.
#
# Below assumes you uploaded a Kaggle Dataset named "oral-cancer-code" with
# your code folder at its root. Replace the name with whatever you used.
!mkdir -p /kaggle/working/code
!cp /kaggle/input/oral-cancer-code/*.py /kaggle/working/code/
!ls /kaggle/working/code
import sys
sys.path.insert(0, "/kaggle/working/code")


# %% [cell] 2: Profile the data
!cd /kaggle/working/code && python profile_data.py --data-root {DATA_ROOT}


# %% [cell] 3: Train one fold (≈ 8-15 min on T4)
# Using 3-fold StratifiedGroupKFold because we have only 5 cancer + 7 healthy
# patients - 5-fold would put 0 cancer patients in some folds.
!cd /kaggle/working/code && python train.py \
    --data-root {DATA_ROOT} \
    --out-dir /kaggle/working/runs \
    --cv sgkf --n-splits 3 --seed 1 \
    --fold 0 --epochs 8 --batch-size 128 --num-workers 4


# %% [cell] 4: Train the remaining folds
import subprocess
for fold in range(1, 3):
    print(f"\n===== FOLD {fold} =====\n")
    subprocess.run([
        "python", "/kaggle/working/code/train.py",
        "--data-root", DATA_ROOT,
        "--out-dir",   "/kaggle/working/runs",
        "--cv", "sgkf", "--n-splits", "3", "--seed", "1",
        "--fold", str(fold),
        "--epochs", "8", "--batch-size", "128", "--num-workers", "4",
    ], check=True)


# %% [cell] 5: Plot learning curves (needed for the presentation)
import json, glob, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for hp in sorted(glob.glob("/kaggle/working/runs/fold*_history.json")):
    h = json.load(open(hp))["history"]
    label = hp.split("/")[-1].replace("_history.json", "")
    ax[0].plot([e["epoch"] for e in h], [e["va_loss"] for e in h], label=label)
    ax[1].plot([e["epoch"] for e in h], [e["va_auc"]  for e in h], label=label)
ax[0].set(title="val loss", xlabel="epoch", ylabel="BCE")
ax[1].set(title="val AUC",  xlabel="epoch", ylabel="AUC")
for a in ax: a.legend(); a.grid(True)
plt.tight_layout(); plt.savefig("/kaggle/working/learning_curves.png", dpi=120)
plt.show()


# %% [cell] 6: Out-of-fold AUC (honest estimate of leaderboard score)
import pandas as pd, glob, numpy as np
from sklearn.metrics import roc_auc_score
oofs = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/working/runs/fold*_oof.csv"))])
print(f"OOF AUC = {roc_auc_score(oofs['y_true'], oofs['y_pred']):.4f}  on {len(oofs)} cells")
print("Per-patient OOF predictions (mean score, true label):")
print(oofs.groupby('patient_id').agg(mean_pred=('y_pred','mean'), label=('y_true','first'))
       .sort_values('mean_pred'))


# %% [cell] 7: Predict on test + write submission.csv
import glob
ckpts = sorted(glob.glob("/kaggle/working/runs/fold*_best.pt"))
print("Using ckpts:", ckpts)
ckpts_str = ' '.join(ckpts)
!cd /kaggle/working/code && python predict.py \
    --data-root {DATA_ROOT} \
    --ckpts {ckpts_str} \
    --out /kaggle/working/submission.csv --tta --batch-size 256 --num-workers 4
!head /kaggle/working/submission.csv
!wc -l /kaggle/working/submission.csv


# %% [cell] 8: Save the notebook and submit
# Click "Save Version" -> "Save & Run All (Commit)". When done, go to the
# "Output" tab -> select submission.csv -> "Submit to Competition".
