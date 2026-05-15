# v15 baseline — evidence used to motivate v16

This folder preserves the v15 training artifacts that the v16 design responds to. The actual model checkpoints (`*_best.pt`, ~90 MB each, ~700 MB total) are **not** included — fetch them from the Kaggle output if you need them. What's here is the lightweight evidence (~10 MB) that documents what v15 actually did.

## Contents

```
.
├── README.md                  <- this file
├── learning_curves.png        <- per-fold val_loss and val_auc over epochs
├── submission_v15.csv         <- the actual submission that scored LB 0.572
└── runs/
    ├── fold{0,1,2}_seed{1,2}_history.json   <- 6 supervised fold runs
    ├── fold{0,1,2}_seed{1,2}_oof.csv        <- per-cell predictions on held-out fold
    └── fulldata_history.json                <- the no-CV "use everything" model
```

## v15 summary numbers

```
Per-fold best val AUC:
  fold0_seed1: 0.9084  at ep 1
  fold0_seed2: 0.8860  at ep 0
  fold1_seed1: 0.7824  at ep 1
  fold1_seed2: 0.8234  at ep 0
  fold2_seed1: 0.8973  at ep 3
  fold2_seed2: 0.8959  at ep 5
  ----------------------------
  CV mean ± std: 0.8656 ± 0.0463

Aggregate OOF (seed-averaged within fold):
  cell-level OOF AUC:    0.8462
  patient-level OOF AUC: 0.9143

LB score (cell-level, public): 0.572
  -> CV-LB gap: 0.29
```

## Why this matters for v16

Two facts from this data drive v16's design:

### 1. The "best_ep" pattern

| fold/seed | best_ep |
|---|---|
| fold0_seed1 | 1 |
| fold0_seed2 | 0 |
| fold1_seed1 | 1 |
| fold1_seed2 | 0 |
| fold2_seed1 | 3 |
| fold2_seed2 | 5 |

Folds 0 and 1 peak at epoch 0–1 then degrade — they're the "hard" folds where the held-out patients are out-of-distribution. Fold 2 peaks late and stays — its held-out patients are in-distribution.

The v15 pipeline picked the best-val-AUC checkpoint per fold/seed. For folds 0 and 1 that's essentially the pre-trained init plus one gradient step. For fold 2 it's a fully trained model. The ensemble averages these inconsistent training stages.

v16 fixes this with **multi-snapshot ensembling at epochs {3, 5}** for *every* fold, plus EMA smoothing. No selection bias, no epoch-mismatch.

### 2. The CV vs LB gap of 0.29

CV mean 0.866 with LB 0.572 means CV is overstating generalization by 0.29 AUC. Two causes, both addressed in v16:

- **3-fold CV groups too many patients per fold** — 2 of 3 folds had test-train-similar patients, hiding the OOD failure. v16 uses **LOPO** (12-fold, single patient per fold) for an honest OOD signal.
- **The supervised stage drifts away from the CoMIR alignment** that SSL learned, because BCE only rewards classification, not cross-modal invariance. v16 adds an **aux NT-Xent loss on frozen CoMIR projections** so the backbone is pulled back toward alignment during fine-tune.

## How to reproduce the numbers above

Numpy-only (no sklearn needed):

```python
import json, csv, numpy as np
from pathlib import Path
from collections import defaultdict

def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2: return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p)+1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), len(y) - pos.sum()
    return (ranks[pos].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

runs = Path("runs")

# Per-fold best:
for p in sorted(runs.glob("fold*_history.json")):
    d = json.load(open(p))
    print(p.stem.replace("_history",""), f"best={d['best_auc']:.4f} at ep {d['best_ep']}")

# Cell-level OOF (seed-averaged):
all_oof = []
for fold in range(3):
    seed_rows = []
    for seed in [1, 2]:
        with open(runs / f"fold{fold}_seed{seed}_oof.csv") as f:
            seed_rows.append([(r["Name"], int(r["patient_id"]), int(r["y_true"]),
                              float(r["y_pred"])) for r in csv.DictReader(f)])
    for i in range(len(seed_rows[0])):
        avg = np.mean([s[i][3] for s in seed_rows])
        all_oof.append(seed_rows[0][i][:3] + (avg,))

y = np.array([r[2] for r in all_oof]); p = np.array([r[3] for r in all_oof])
print(f"cell-level OOF AUC: {auc(y, p):.4f}")

# Patient-level OOF:
pat = defaultdict(list)
for _, pid, yt, yp in all_oof:
    pat[pid].append((yt, yp))
pp_y = np.array([rows[0][0] for rows in pat.values()])
pp_p = np.array([np.mean([r[1] for r in rows]) for rows in pat.values()])
print(f"patient-level OOF AUC: {auc(pp_y, pp_p):.4f}")
```
