# Multimodal cancer-cell classification

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Per-cell malignant/benign classification of oral-cancer cells imaged in two paired modalities, brightfield (BF) and fluorescence (FL), at 128×128 grayscale. The dataset is small where it matters: 12 training patients, about 114,000 labeled cells, and a 59,000-cell test set whose patients never appear in training.

With only 12 patients the model is data-bound, not capacity-bound. The gains here came from growing the effective training set with semi-supervised distillation and from diagnosing failures, not from bigger networks. Public-leaderboard AUC went from 0.7455 (the first solid supervised model) to 0.8392 (best single model). The private split decides the final score.

![Public LB progression](presentation/figures/lb_progression.png)

## Results

| Version | Change | Reference | Public LB |
|---|---|---|---|
| v19 | Supervised baseline: dual EfficientNet-B0, per-patient MIL aux loss, AdaBN, test-set stain norm, TTA | — | 0.7455 |
| v41 | Added label smoothing, dropout, RandomResizedCrop, multiscale TTA | — | 0.7563 |
| v44 | Hard pseudo-labels: ~9,400 confident test cells added to training | Lee 2013 | 0.7812 |
| v46 | Soft-target distillation: all 59,040 test cells, teacher probability as the target | Hinton 2015 | 0.8236 |
| v47 | Second distillation round (teacher swapped to v46) | Xie 2020 | 0.8264 ens / 0.8355 best seed |
| v58 | Intermediate co-attention fusion, single model | Lian 2024 | 0.8392 (best) |

The largest single step was hard-to-soft pseudo-labels at v46. Hard thresholding keeps only confident cells and throws away 84% of the test set; soft distillation keeps all 59k cells, each weighted by the teacher's probability. That change alone added about +0.042. A second distillation round (v47) added almost nothing on the mean. Later, intermediate co-attention fusion (v58) raised the best single model to 0.8392. Larger backbones (EfficientNetV2-S, 0.7376) and higher input resolution (192px, 0.7823) both made it worse, which is the data-bound point again.

Each headline gain is scored against the seed-to-seed noise floor, measured from three seeds per recipe. The soft-pseudo gain sits about 4 standard errors above that floor; the second distillation round is about 0.3, so it is not distinguishable from noise. A per-cell check found that 97% of the disagreement between the best seed and the seed average falls in the teacher's uncertain middle band, which is evidence that the best-seed result is signal rather than a lucky split.

## What didn't work

The failures were as useful as the wins:

- **Self-supervised pretraining (v42)** collapsed to 0.59. With one patient per cell, a cross-modal contrastive task can be solved by recognizing the patient rather than the cell, which is the exact shortcut a patient-disjoint test set punishes.
- **Stacking four changes at once (v43)** regressed and left no way to tell which change was responsible. After that I changed one or two related things per submission.
- **Ensembling gave no lift beyond noise.** A dozen blends — seed averaging, cross-recipe averaging, a decorrelated feature-GBM — all scored below the best single model. The lone exception was a weighted blend of the two best, near-tied, different-recipe models (0.0037 apart): it reached 0.8404 against the best single's 0.8392, but +0.0012 is about 0.1σ, a statistical tie. With 12 patients the seed spread (~0.02 AUC) is wide enough that averaging only pulls toward the mean, so a careful blend and the best single model are interchangeable.

## Method

Two EfficientNet-B0 branches, one per modality, with late concat fusion and a small MLP head. The parts that matter sit outside the backbone: a per-patient multiple-instance auxiliary loss, AdaBN and test-set stain normalization to handle the train/test brightness gap, and 40-way test-time augmentation. Distillation versions add the teacher's predictions on the test set to the training targets.

![Architecture](presentation/figures/arch_diagram.png)

## Repository

| Path | Contents |
|---|---|
| `notebooks/` | One source notebook per version (`improvedvNN_source.ipynb`), runnable end to end on Kaggle (T4 ×2). |
| `presentation/` | Slide deck (`A3_cancer_challenge.pptx` / `.pdf`) and the figures shown above. |
| `overleaf-report/figure-sources/` | Scripts that build the figures and the noise-floor table from the prediction CSVs. |
| `LB_HISTORY.md` | Every submission in order: the change, the public LB, and the one-line lesson. |
| `results/` | Per-version submission CSVs. Training checkpoints are gitignored. |

## Reproducing the analysis (CPU only)

```bash
pip install -r requirements.txt
python overleaf-report/figure-sources/build_lb_progression.py        # LB progression chart
python overleaf-report/figure-sources/build_noise_floor_analysis.py  # per-seed significance table
```

Training runs are Kaggle notebooks. Open a source notebook, attach the competition data (and a teacher-prediction dataset for the distillation versions), and run all. A distillation version takes about 5–6 hours on one T4 ×2 session.

## Context

Course project for Uppsala University 1MD042 (Advanced Deep Learning for Image Processing), spring 2026. The grade is based on method and presentation rather than leaderboard performance, which is why the failure analysis gets as much space as the results.

## License

MIT, see [`LICENSE`](LICENSE). The reference papers and the competition dataset are not included; see [`REFERENCES.md`](REFERENCES.md) for citations.
