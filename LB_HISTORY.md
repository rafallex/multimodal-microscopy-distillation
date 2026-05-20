# Leaderboard history

Quick reference for every Kaggle submission run, in chronological order.
The corresponding source notebooks live in `notebooks/improvedvNN_source.ipynb`.

| Version | Backbone | Input | Recipe / key change | Public LB | Notes |
|---|---|---|---|---|---|
| baseline | ResNet-18 | 128 native | starter notebook | — | `oral_cancer_baseline.ipynb` |
| v15 | ResNet-18 | 128 native | early MIL baseline | — | `v15_baseline/` |
| v16 | ResNet-18 | 128 native | LOPO CV + EMA-smoothed multi-snapshot ensemble | — | `v16_results/` |
| v17 | EffNet-B0 | 128 native | return to v11 baseline + pseudo-labels | — | `notebooks/improvedv17_source.ipynb` |
| v18 | EffNet-B0 | 128 native | v17 floor recipe | — | `notebooks/improvedv18_source.ipynb` |
| **v19** | **EffNet-B0** | **128 native** | **MIL + strong aug + AdaBN + 8-way D4 TTA + test stain norm** | **0.7455** | **`notebooks/improvedv19_source.ipynb` — best single-model so far** |
| v20 | EffNet-B0 | 128 native | new aug recipe (failed) | 0.5974 | abandoned |
| v21 | ResNet-50 | **224 upscale** | first ResNet-50 attempt (confounded with upscale) | 0.7018 | Lu 2020: "no interpolation" warning |
| v22 | (v19 + v21) | — | local CSV ensemble (sigmoid_avg / rank_avg / geomean) | 0.7422 / 0.7436 / 0.7328 | all underperformed v19 alone |
| v23 | EffNet-B0 | 128 native | v19 recipe, BASE_SEED=2 | 0.7154 | 0.03 LB spread from seed alone |
| v27 | EffNet-B0 | 128 native | v19 + 2-patient val holdout + disc LR + best-val ckpt | — | `best_epoch=0` collapse, val patients adversarial |
| v30 | EffNet-B0 | 128 native | v19 + disc LR only (no val), backbone_lr_ratio=0.1 | 0.6448 | disc LR at 0.1 ratio HURT by ~0.10 |
| **v34** | **ResNet-50** | **128 native** | **v19 recipe, ResNet-50 backbone, no upscale, no disc LR** | **0.7155** | **`notebooks/improvedv34_source.ipynb` — teacher's backbone tested cleanly** |
| v35 | DenseNet-201 | 128 native | v19 recipe, DenseNet-201 backbone | (queued) | `notebooks/improvedv35_source.ipynb` |

## Top-of-leaderboard reference

LB leader sits at ~0.7832 (per Kaggle competition page).

## Lessons by experiment

| Experiment | Lesson |
|---|---|
| v19 vs v23 (seed diversity) | Same recipe gives 0.03 LB spread per seed — high-variance regime |
| v21 (ResNet-50 + 224 upscale) | Confounded change. Lu 2020's "no interpolation" warning matters at this image scale |
| v22 ensembles (v19 + v21) | Weaker member dragged the ensemble below v19 alone. Don't ensemble when components are >0.02 apart |
| v27 (val patient holdout) | Single 2-patient holdout on N=12 is fundamentally too noisy. pat_5 (saturated FL) + pat_14 (dark FL) are at FL exposure extremes — adversarial val choice. best_epoch=0 = essentially untrained model |
| v30 (discriminative LR) | Standard "1/10 of original LR" recipe (L4 slide 96) is too aggressive for transferring ImageNet features to microscopy. Backbone needs more LR to learn texture features the ImageNet weights didn't see |
| **v34 (ResNet-50 clean test)** | **ResNet-50 at 128 native (no upscale, no disc LR) ≈ EffNet-B0 different-seed result (v23). The teacher's recommended backbone works — but doesn't beat v19. v21's 0.7018 was due to the 224 upscale, not ResNet-50 itself.** |

## Final-submission selection strategy

Kaggle lets you pick 2 submissions for private LB. Plan:

1. **v19 (LB 0.7455)** — known safety net.
2. **Best of {v34, v35}** — your "tested teacher's recommended backbones cleanly" submission.

If v35 underperforms v34, pick v34. If both underperform v19, pick v19 twice — you've still demonstrated the experiment honestly in the report.

## What's NOT tracked here

- The competition dataset (`multimodal-cancer-classification-challenge-2026/`) is gitignored.
- Per-version result artifacts (.pt checkpoints + extracted result folders) live in the
  local-only `results/vNN/` tree. The `submission.csv`, `history.json`, and `learning_curves.png`
  are kept locally for offline analysis but excluded from git.
- A more detailed local-only summary lives at `results/LB_SUMMARY.md` (also gitignored).
