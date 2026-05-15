# A3-ADL — Multimodal Cancer Classification Challenge 2026

Coursework for **Advanced Deep Learning for Image Processing (1MD042)** at Uppsala University, spring 2026.

Binary cell-level classification (cancer vs healthy) from paired **brightfield (BF) + fluorescence (FL)** microscopy images at 128×128 grayscale. **12 train patients with leave-one-patient-out CV**; test patients are unknown — out-of-distribution generalization is the dominant failure mode.

---

## Results so far

| Version | Method headline | CV AUC | Cell OOF | Patient OOF | LB |
|---|---|---|---|---|---|
| `oral_cancer_baseline` | Course starter (ResNet-18 multimodal) | — | — | — | — |
| `improvedv10` – `v13` | Iterative tweaks: AdamW, OneCycleLR, paired aug, patient-balanced sampler | rising | — | — | — |
| `improvedv14` | First mixup attempt | — | — | — | — |
| `improvedv15` | **CoMIR-style contrastive SSL** + discriminative LR (1:10) | **0.866 ± 0.046** | 0.846 | 0.914 | **0.572** |
| `improvedv16` | 9 changes addressing v15's OOD collapse (see below) | TBD | TBD | TBD | TBD |

**The v15 CV ↔ LB gap (0.866 → 0.572) is the entire problem v16 was designed to solve.** Two of the three v15 CV folds had val AUC > 0.88; the third had 0.78–0.82 and peaked at epoch 0–1 (overfitting to train-distribution stains immediately). v15 saved the *best-AUC* epoch per fold, which on the hard folds was epoch 0 — essentially the pre-trained initialization plus one gradient step. See `v15_baseline/learning_curves.png` for the visual evidence.

---

## v16 — nine changes from v15

Each change has a one-line motivation. Full reasoning lives in the title cell of `improvedv16_source.ipynb`.

| # | Change | Why |
|---|---|---|
| 1 | **No mixup** during supervised fine-tune | Mixup fought CoMIR's BF↔FL alignment (synthetic mixtures have no real cross-modal correspondence) |
| 2 | **LOPO cross-validation** (12 folds, one patient out, single seed) | Honest OOD estimator; 3-fold stratified-group CV was hiding the gap |
| 3 | **Heavier stain aug during CoMIR SSL** (ColorJitter 0.5 + RandomGamma γ∈[0.7,1.4]) | The contrastive task must learn content-correspondence *despite* arbitrary stain variance |
| 4 | **Logit-space TTA averaging** (sigmoid once at the end) | Preserves dynamic range across 8 D4 augmentations |
| 5 | **Multi-snapshot ensemble** rank-averaged with mean-of-logits | Robust to per-model calibration drift; no val_auc gate (LOPO doesn't give one) |
| 6 | **Label smoothing ε=0.05** on BCE targets | L4 lecture p.48; reduces logit saturation so the two ensembling schemes agree |
| 7 | **Snapshots at epochs {3, 5}** per fold | v15 evidence: hard folds peak early, easy folds peak late — the ensemble decides without selection |
| 8 | **EMA of supervised weights, decay 0.99** | Tuned for our 426-step training length (`1/(1-decay)` window ≈ 100 steps, not the conventional 1000) |
| 9 | **Aux NT-Xent alignment loss** with frozen CoMIR projection heads | Anchors the backbone to the SSL objective during BCE fine-tune; directly opposes the OOD drift |

Plus a free hedge: `submission_patient_agg.csv` replaces each cell's prediction with its patient's mean, in case the LB scores patient-level rather than cell-level.

---

## v15 evidence that motivates v16

From `v15_baseline/runs/*_history.json`:

```
fold0_seed1: best_auc=0.9084 at ep 1   <- peak immediately, then declines
fold0_seed2: best_auc=0.8860 at ep 0   <- peak at init
fold1_seed1: best_auc=0.7824 at ep 1   <- weakest fold, peaks early
fold1_seed2: best_auc=0.8234 at ep 0   <- peak at init
fold2_seed1: best_auc=0.8973 at ep 3   <- stable, late peak
fold2_seed2: best_auc=0.8959 at ep 5   <- stable, latest peak
```

**Diagnosis**: median best_ep is 1 (out of 8 trained). Folds 0 and 1 hold out OOD-like patients and overfit immediately; fold 2 holds out in-distribution-like patients and trains normally. v15 saved best-val-AUC per fold, which on the hard folds is essentially the initialization. v16's LOPO scheme + multi-snapshot ensembling at {3, 5} captures both trajectory regimes without having to choose.

From `v15_baseline/runs/fold*_seed*_oof.csv` aggregated against true labels:

```
Per-patient OOF mean prediction (sorted):
  pat 7 (y=0): mean_pred=0.072
  pat10 (y=0): mean_pred=0.138
  pat13 (y=0): mean_pred=0.219
  pat11 (y=0): mean_pred=0.268
  pat15 (y=0): mean_pred=0.286
  pat14 (y=0): mean_pred=0.474   <- ambiguous, OOD-like
  pat17 (y=1): mean_pred=0.498   <- positive patient pulled down — failure case
  pat 5 (y=1): mean_pred=0.543
  pat16 (y=1): mean_pred=0.633
  pat 9 (y=0): mean_pred=0.665   <- negative patient pulled up — failure case
  pat 3 (y=1): mean_pred=0.839
  pat18 (y=1): mean_pred=0.975
```

Patients **9** and **17** are the OOD failures. v16's LOPO scheme will expose patient-by-patient how each is mispredicted; the aux NT-Xent + heavier stain aug should reduce these.

---

## Repo layout

```
.
├── README.md                              <- this file
├── .gitignore
├── code/                                  <- modular Python pipeline (early baseline)
│   ├── README.md
│   ├── dataset.py, model.py, transforms.py, splits.py
│   ├── train.py, predict.py, profile_data.py
│   └── kaggle_notebook_cells.py
├── oral_cancer_baseline.ipynb             <- course starter
├── improvedvN_source.ipynb                <- editable Kaggle source (v10..v16)
├── improvedvN_run.ipynb                   <- committed run output with cell outputs
├── improvedv13_colab_source.ipynb         <- Colab variant of v13
├── improvedv13_colab_run.ipynb
├── improvedv16_source.ipynb               <- CURRENT - Kaggle commit target
├── improvedv16_colab_source.ipynb         <- Colab port (gdown-based, no drive.mount)
└── v15_baseline/                          <- v15 evidence used to motivate v16
    ├── README.md
    ├── learning_curves.png
    ├── submission_v15.csv
    └── runs/                              <- history JSONs + OOF CSVs (no .pt files)
```

**Conventions:**
- `*_source.ipynb` — clean editable notebook (small file size, no cell outputs)
- `*_run.ipynb` — committed run with cell outputs and metadata (large file size)
- `*_colab_*.ipynb` — Colab variant with Google Drive paths instead of Kaggle paths

Data (the 770 MB competition zip) and trained checkpoints (~90 MB each) are gitignored — fetch them from the Kaggle competition page or from Google Drive (`gdown` flow described in `improvedv16_colab_source.ipynb`).

Documentation:

- [`docs/methods-diagram.md`](docs/methods-diagram.md) — visual pipeline diagram, architecture diagram, problem → fix mapping, and a Gantt chart of the compute timeline. **Start here** for the 60-second overview.
- [`docs/notebook-walkthrough.md`](docs/notebook-walkthrough.md) — how to read the 14 notebooks: which version contributed what, which cells in v16 are worth focusing on, what the `_source` / `_run` / `_colab` suffix conventions mean.
- [`docs/design-decisions.md`](docs/design-decisions.md) — three formal ADRs covering the v15 → v16 design choices, plus supporting numerical audits (lecture-tip mining, AMP correctness, EMA decay math).
- [`code/README.md`](code/README.md) — the modular Python pipeline from the early baseline (parallel implementation, not used by v16 itself).
- [`v15_baseline/README.md`](v15_baseline/README.md) — what's in the v15 evidence folder and how to reproduce the OOF numbers.

---

## Running v16

### On Kaggle (primary target)

1. Open `improvedv16_source.ipynb` in the Kaggle notebook editor
2. **Settings**: Accelerator = `GPU T4 ×2`, Internet = `On`
3. **Input**: attach `multimodal-cancer-classification-challenge-2026` so it mounts at `/kaggle/input/competitions/multimodal-cancer-classification-challenge-2026/`
4. `Save Version → Save & Run All`
5. After ~8h 15min, download `submission.csv` from the output tab and submit to the competition

### On Colab (fallback for when Kaggle GPU quota is exhausted)

1. Open `improvedv16_colab_source.ipynb` in VSCode connected to a Colab T4 runtime
2. In Google Drive, share `multimodal-cancer-classification-challenge-2026.zip` as **"Anyone with the link" → Viewer**
3. Edit the `FILE_ID` constant in the first cell
4. `Run All`. Outputs land in `/content/v16_runs/` (ephemeral — download before the runtime ends)

### Compute budget (Kaggle T4×2, only GPU 0 used)

| Stage | Time |
|---|---|
| Cache (one-time per session) | ~45 min |
| CoMIR SSL (5 epochs, batch 256, with projection heads) | ~25 min |
| LOPO supervised (12 folds × 6 epochs) | ~4h 20min |
| Full-data × 2 seeds | ~44 min |
| 28-way TTA inference (24 LOPO + 4 full-data snapshots × 8 D4 augs) | ~2h |
| **Total** | **~8h 15min** (well under 9h commit limit) |

---

## Citations

1. Pielawski, N., Wetzer, E., Öfverstedt, J., Lu, J., Wählby, C., **Lindblad, J.**, & Sladoje, N. (2020). *CoMIR: Contrastive Multimodal Image Representation for Registration.* NeurIPS 2020. — The SSL pretext task underlying v15 / v16.
2. Smith, L. N., & Topin, N. (2019). *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.* — OneCycleLR.
3. Polyak averaging / EMA: see e.g. He et al. (2020) *Momentum Contrast for Unsupervised Visual Representation Learning.*

L4 lecture (Uppsala 1MD042) provided the discriminative LR ratio (1:10 backbone:head, p.96) and label smoothing motivation (p.48).

---

*Repository maintained for coursework. Not affiliated with the competition organizers.*
