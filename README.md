# Multimodal cancer-cell classification

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Public-leaderboard AUC 0.7455 → 0.8236 on a 12-patient, patient-disjoint oral-cancer benchmark, by growing the effective training set with soft-target knowledge distillation (mean of three seeds).**

Per-cell malignant/benign classification of oral-cancer cells imaged in two paired modalities, brightfield (BF) and fluorescence (FL), at 128×128 grayscale. The dataset is small: 12 training patients, about 114,000 labelled cells, and a 59,000-cell test set whose patients never appear in training.

With only 12 patients the model is data-bound rather than capacity-bound, so the gains came from growing the effective training set with semi-supervised distillation, not from bigger networks. Public-leaderboard AUC went from 0.7455 (the first solid supervised model) to 0.8236 (soft-target distillation), averaged over three seeds. The private split decides the final score.

## What this project demonstrates

- **Statistical rigour:** every headline gain is validated against a measured seed-to-seed noise floor, separating real improvement from run-to-run variance.
- **Systematic ablation:** one change at a time across 25+ versions, with each failure diagnosed before the next step.
- **Semi-supervised learning:** hard pseudo-labelling (Lee 2013) and soft-target knowledge distillation (Hinton 2015) to exploit the unlabelled test images.
- **Multimodal modelling and domain adaptation:** a dual-branch backbone with late fusion, a per-patient multiple-instance loss, and test-time adaptation (AdaBN, stain normalisation) for a patient-shifted test set.
- **Honest failure analysis:** see [`LB_HISTORY.md`](LB_HISTORY.md) for the full experiment log, including the approaches that did not work and why.

![Public LB progression](presentation/figures/lb_progression.png)

## Results

| Version | Change | Reference | Public LB |
|---|---|---|---|
| v19 | Supervised baseline: dual EfficientNet-B0, per-patient MIL aux loss, AdaBN, test-set stain norm, TTA | — | 0.7455 |
| v41 | Added label smoothing, dropout, RandomResizedCrop, multiscale TTA | — | 0.7563 |
| v44 | Hard pseudo-labels: ~9,400 confident test cells added to training | Lee 2013 | 0.7812 |
| v46 | Soft-target distillation: all 59,040 test cells, teacher probability as the target | Hinton 2015 | 0.8236 |

The largest single step was the move from hard to soft pseudo-labels at v46. Hard thresholding keeps only confident cells and discards 84% of the test set; soft distillation keeps all 59k cells, each weighted by the teacher's probability. That change added about +0.042, the biggest jump in the project. Every gain traced to growing the effective training set rather than to bigger networks.

Each headline gain is scored against the seed-to-seed noise floor, measured from three seeds per recipe. The soft-distillation gain (v44→v46) sits about four standard errors above that floor, the one clearly significant jump in the project.

## Method

Two EfficientNet-B0 branches, one per modality, with late concat fusion and a small MLP head. The parts that matter sit outside the backbone: a per-patient multiple-instance auxiliary loss, AdaBN and test-set stain normalisation to handle the train/test brightness gap, and 40-way test-time augmentation. Distillation versions add the teacher's predictions on the test set to the training targets.

![Architecture](presentation/figures/arch_diagram.png)

## Repository

**Start here:** [`notebooks/improvedv46_source.ipynb`](notebooks/improvedv46_source.ipynb) is the final-model recipe (v46, soft-target distillation, public LB 0.8236). `notebooks/` holds one source notebook per version; read [`LB_HISTORY.md`](LB_HISTORY.md) first for the version-by-version story, and see the report ([`overleaf-report/report.tex`](overleaf-report/report.tex)) for the learning curves and full results.

| Path | Contents |
|---|---|
| `overleaf-report/report.tex` | Full methodology write-up (LaTeX, Overleaf-ready) with `references.bib` and `figures/`. |
| `notebooks/` | One source notebook per version (`improvedvNN_source.ipynb`), runnable on Kaggle (T4 ×2). |
| `presentation/figures/` | LB-progression and architecture figures used above. |
| `overleaf-report/figure-sources/` | Scripts that build the figures from the prediction CSVs. |
| `LB_HISTORY.md` | Every submission in order: the change, the public LB, and the one-line lesson. |
| `results/` | Per-version submission CSVs. Training checkpoints are gitignored. |

## Reproducing the figures (CPU only)

```bash
pip install -r requirements.txt
python overleaf-report/figure-sources/build_lb_progression.py         # LB progression chart
python overleaf-report/figure-sources/build_arch_diagram.py           # architecture diagram
python overleaf-report/figure-sources/build_teacher_prob_histogram.py # teacher-probability histogram
```

Training runs are Kaggle notebooks. Open a source notebook, attach the competition data (and a teacher-prediction dataset for the distillation versions), and run all. A distillation run takes about 5–6 hours on one T4 ×2 session.

## Context

Course project for Uppsala University 1MD042 (Advanced Deep Learning for Image Processing), spring 2026. Author: Rafael Tavares Proença.

## License

MIT, see [`LICENSE`](LICENSE). The reference papers and the competition dataset are not included; see [`REFERENCES.md`](REFERENCES.md) for citations.
