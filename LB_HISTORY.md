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
| v19 | EffNet-B0 | 128 native | MIL + strong aug + AdaBN + 8-way D4 TTA + test stain norm | 0.7455 | `notebooks/improvedv19_source.ipynb` — long-time best, now safety net |
| v20 | EffNet-B0 | 128 native | new aug recipe (failed) | 0.5974 | abandoned |
| v21 | ResNet-50 | **224 upscale** | first ResNet-50 attempt (confounded with upscale) | 0.7018 | "no interpolation" warning from the literature |
| v22 | (v19 + v21) | — | local CSV ensemble (sigmoid_avg / rank_avg / geomean) | 0.7422 / 0.7436 / 0.7328 | all underperformed v19 alone |
| v23 | EffNet-B0 | 128 native | v19 recipe, BASE_SEED=2 | 0.7154 | 0.03 LB spread from seed alone |
| v27 | EffNet-B0 | 128 native | v19 + 2-patient val holdout + disc LR + best-val ckpt | — | `best_epoch=0` collapse, val patients adversarial |
| v30 | EffNet-B0 | 128 native | v19 + disc LR only (no val), backbone_lr_ratio=0.1 | 0.6448 | disc LR at 0.1 ratio HURT by ~0.10 |
| v34 | ResNet-50 | 128 native | v19 recipe, ResNet-50 backbone, no upscale, no disc LR | 0.7155 | `notebooks/improvedv34_source.ipynb` — ResNet-50 backbone tested cleanly |
| v35 | DenseNet-201 | 128 native | v19 recipe, DenseNet-201 backbone | (queued) | `notebooks/improvedv35_source.ipynb` |
| v36 | SE-ResNet-50 | 128 native | v34 recipe + SE channel attention (timm seresnet50) + 20 epochs | (not submitted) | `notebooks/improvedv36_source.ipynb` |
| v37 | EffNet-B0 | 128 native | v19 + a published modality-specific augmentation recipe (heavy FL color jitter brightness/contrast=0.8) | 0.7092 | `notebooks/improvedv37_source.ipynb` — that FL aug finding is multi-channel specific, didn't transfer to grayscale |
| v38 | EffNet-B0 | 128 native | v19 + input-level (early) fusion (single 2-ch backbone instead of dual late-fusion) | 0.7147 | `notebooks/improvedv38_source.ipynb` — the early-fusion finding didn't transfer to grayscale 128 |
| **v41** | **EffNet-B0** | **128 native** | **v19 + label smoothing 0.05 + multiscale TTA (3 scales × 8 D4) + dropout 0.4 + paired RandomResizedCrop** | **0.7563** | **`notebooks/improvedv41_source.ipynb` — NEW BEST. Four regularizers stacked cleanly: +0.011 over v19** |
| v42 | EffNet-B0 | 128 native | v19 + cross-modal contrastive InfoNCE SSL pretrain (10 ep) → finetune (8 ep @ LR 1e-4) | 0.5908 | `notebooks/improvedv42_source.ipynb` — catastrophic. SSL learned patient identity, not cell content |
| v43 | EffNet-B0 | 128 native | v41 + 3-seed ensemble + SWA(last 4 ep) + FL-tuned aug (CJ 0.5/0.3 + RandomGamma) + WD 1e-4→3e-4 + 40-way TTA (5 scales) | 0.7444 | `notebooks/improvedv43_source.ipynb` — **REGRESSION −0.012 vs v41**. The stacked extras hurt without pseudo cells in the loss |
| **v44** | **EffNet-B0** | **128 native** | **v43 recipe + pseudo-labels from v41 at threshold 0.05/0.95 (~9.3k confident test cells added with patient_id=−1, MIL skips them)** | **0.7812** | **`notebooks/improvedv44_source.ipynb` — NEW BEST. +0.025 over v41. Pseudo-labels carried the win** |
| v45_probe | (v41 + v44) | — | local CSV ensemble (sigmoid_avg) | 0.7729 | underperformed v44 alone by −0.008 but added +0.004 over the naive LB-average of members. Pearson(v41,v44)=0.936. Gap of 0.025 LB between members was too wide to gain — same pattern as v22, gentler magnitude |
| **v44_seed1** | **EffNet-B0** | **128 native** | **single-seed extraction from v44's 3-seed ensemble (submission_seed1.csv)** | **0.7844** | **NEW BEST. +0.0032 over the v44 ensemble. Seed 1 of v44 happened to be the lucky one. Confirms the seed-luck hypothesis at this LB regime. v44_seed1 is now the distillation source for v46** |
| **v45** | **EffNet-B0** | **128 native** | **v44 minus FL-tuned aug + WD reverted 3e-4→1e-4 + pseudo source upgraded v41→v44 (self-training iteration). Keeps 3-seed × SWA, 40-way TTA, paired RRC, label smoothing, dropout 0.4** | (queued) | **`notebooks/improvedv45_source.ipynb` — requires `submissionv44` Kaggle dataset (upload v44's submission.csv as a new private Kaggle dataset). Target ~0.79–0.80 LB.** |
| **v46** | **EffNet-B0** | **128 native** | **v44 stripped (no FL aug, WD 1e-4) + SOFT pseudo-labels from v44_seed1 (the lucky teacher): all 59k test cells, raw probs as BCE targets (Hinton 2015 distillation), pseudo loss weighted 0.5** | **0.8236** | **`notebooks/improvedv46_source.ipynb` — NEW BEST. +0.039 over v44 seed1, +0.042 over v44 ensemble. Distillation exceeded my forecast of +0.005 to +0.015. All 3 seeds reached tr_auc ≈ 0.993 (vs v44's 0.989). Need +0.0264 more to reach 0.85** |
| v46_seed1 | EffNet-B0 | 128 native | single-seed extract from v46 (submission_seed1.csv) | 0.8157 | v46 seed1 alone scored 0.0079 below the v46 ensemble |
| v46_seed2 | EffNet-B0 | 128 native | single-seed extract from v46 (submission_seed2.csv) | 0.8229 | v46 seed2 alone scored 0.0007 below the v46 ensemble. **Different pattern from v44**: in v46 the ensemble beat both observed seeds, meaning within-recipe averaging genuinely added signal here (consistent with the tighter tr_auc convergence ≈ 0.993 across seeds vs v44's 0.989 spread) |

_This project's reported work ends at v46._

## Headline LB milestones (this project)

| Version | Public LB | Note |
|---|---|---|
| v19 | 0.7455 | supervised floor |
| v44 | 0.7812 | hard pseudo-labels |
| v46 | 0.8236 | soft-target distillation |

(Own scores only.)

## Lessons by experiment

| Experiment | Lesson |
|---|---|
| v19 vs v23 (seed diversity) | Same recipe gives 0.03 LB spread per seed — high-variance regime |
| v21 (ResNet-50 + 224 upscale) | Confounded change. The "no interpolation" warning from the literature matters at this image scale |
| v22 ensembles (v19 + v21) | Weaker member dragged the ensemble below v19 alone. Don't ensemble when components are >0.02 apart |
| **cross-recipe ensemble rule (v22 + v45_probe)** | **Two attempts at cross-recipe sigmoid-averaging have both regressed below the stronger member: v22 (v19+v21, 0.05 LB gap) and v45_probe (v44+v41, 0.025 LB gap). The +0.004 lift v45_probe added over the naive LB-average suggests cross-recipe ensembling does add small signal, but the LB gap penalty dominates for any practical gap. Operational rule: stop trying cross-recipe ensembles. Within-recipe seed + SWA averaging (which is what v44 is doing internally) remains the winning ensemble pattern.** |
| v27 (val patient holdout) | Single 2-patient holdout on N=12 is fundamentally too noisy. pat_5 (saturated FL) + pat_14 (dark FL) are at FL exposure extremes — adversarial val choice. best_epoch=0 = essentially untrained model |
| v30 (discriminative LR) | Standard "1/10 of original LR" recipe is too aggressive for transferring ImageNet features to microscopy. Backbone needs more LR to learn texture features the ImageNet weights didn't see |
| v34 (ResNet-50 clean test) | ResNet-50 at 128 native (no upscale, no disc LR) ≈ EffNet-B0 different-seed result (v23). The ResNet-50 backbone works — but doesn't beat v19. v21's 0.7018 was due to the 224 upscale, not ResNet-50 itself. |
| v36 (SE-ResNet-50) | SE attention (Hu et al. 2018) added to every bottleneck. EfficientNet has SE built-in; v34 did not. v36 isolates whether channel attention closes the gap. +8 epochs (12→20) since larger model needs more training steps. Not submitted. |
| v37 (modality-specific aug) | A published ablation showed FL color jitter alone is worth ~19 pp F1 on that setup. Didn't transfer — their FL is 4-channel emission stack, ours is 1-channel grayscale; the strong brightness/contrast=0.8 jitter is washing out the only intensity signal we have. |
| v38 (early fusion) | A published result reported early fusion beating late fusion by ~1.7 pp F1 on that data. Didn't transfer to grayscale 128. Likely test-set BF/FL alignment isn't tight enough for pixel-level fusion to win over feature-level fusion. |
| **v41 (stacked regularizers)** | **Four cheap textbook additions (label smoothing 0.05 + dropout 0.3→0.4 + paired RandomResizedCrop + 24-way multiscale TTA) stacked roughly additively for +0.011 over v19. Lesson: textbook regularization still has headroom even after a strong baseline. The path forward is stacking small, well-motivated regularizers, not architectural overhauls.** |
| v42 (cross-modal contrastive SSL) | InfoNCE loss converged to 0.02 (suspiciously low — log(128)≈4.85 is random) in epoch 1, then plateaued. Failure mode: with only one patient per cell, paired BF/FL of cell *i* always share a patient and unpaired BF/FL of cell *j*≠*i* usually don't, so encoding *patient identity* solves the contrastive task at ≥95%. Backbone got optimized to *strengthen* patient signature; finetune memorized it; test patients are different patients → tr_auc 0.96 vs LB 0.59. Cross-modal SSL needs positive pairs that span patient boundaries to work as a patient-shortcut remover. |
| v43 standalone (stack-on-v41) | v41 was at a local max. Stacking FL-tuned aug + WD 1e-4→3e-4 + wider TTA + 3-seed/SWA ensemble all at once produced LB 0.7444 — a −0.012 regression. Lesson: don't compose 4 unvalidated changes in one experiment; at least one of them was a net negative and the others didn't cover for it. The seed ensemble + SWA didn't even rescue the regression. |
| **v44 (pseudo carries everything)** | **v44 = v43 + pseudo-labels (only diff). v44 (0.7812) beat v43 (0.7444) by +0.037 and v41 (0.7563) by +0.025. Arithmetic: pseudo-labels alone are worth ≥+0.037 LB, because they also had to overcome v43's −0.012 negative stack. Lee 2013-style confidence-thresholded pseudo-labeling is dramatically more valuable than any combination of textbook regularizers when the distillation source is decent (v41 at 0.7563 was enough). The bottleneck wasn't the loss or the augmentation — it was the size of the labeled training set.** |

## Final submission

Kaggle lets you select up to two submissions for the private leaderboard. The reported final model is **v46 (soft-target distillation, LB 0.8236)** — the variance-reduced 3-seed ensemble, which beat all three of its individual seeds.

On 12 patients the per-seed spread is wide (~0.02 AUC), so averaging the seeds reduces variance without surrendering signal, which is the safer bet for the hidden patient split than any single lucky seed. Cross-recipe blends were not worth a slot: every wide-gap blend tried (v22, v45_probe) regressed below its stronger member, outside the ~0.015 gap rule (M2).

## What's NOT tracked here

- The competition dataset (`multimodal-cancer-classification-challenge-2026/`) is gitignored.
- Per-version result artifacts (.pt checkpoints + extracted result folders) live in the
  local-only `results/vNN/` tree. The `submission.csv`, `history.json`, and `learning_curves.png`
  are kept locally for offline analysis but excluded from git.
- A more detailed local-only summary lives at `results/LB_SUMMARY.md` (also gitignored).

## Meta-lessons (cross-experiment synthesis)

Five insights that recur across multiple experiments and that drove the final-version recipe choices. These consolidate the per-experiment lessons above.

### M1. More data helped far more than a better model

Between v19 and v41 I gained +0.011 LB from four textbook regularizers stacked cleanly (label smoothing, dropout 0.4, paired RandomResizedCrop, multiscale TTA). Between v41 and v46 I gained +0.067 LB by growing the labeled training set: first via hard pseudo-labels (+0.025) and then via soft-target distillation that used 6× more pseudo cells (+0.039 additional). On 12 training patients with ~10k cells each, the bottleneck was the size and quality of the supervisory signal, not the model's capacity.

### M2. Ensembling only reduced variance, it did not add signal

On 12 patients the seed spread is wide, so ensembling buys variance reduction, not signal. v46's 3-seed mean beat all three of its individual seeds, so within-recipe averaging is the pattern that held. Cross-recipe sigmoid averaging at wide gaps failed twice: v22 (v19+v21, 0.05 gap) and v45_probe (v41+v44, 0.025 gap) both regressed below the better member. That gives the operational rule: don't average across recipes if the LB gap is wider than ~0.015. Net: no ensemble gave a lift beyond noise; averaging only reduces variance.

### M3. Patient identity was the main shortcut the model wanted to learn

The training data has 12 patients, the test set has disjoint patients, and per-patient FL exposure varies by ~20% mean intensity. Any method that can encode patient identity will preferentially learn it. v42 confirmed this catastrophically: cross-modal SSL on paired (BF, FL) of the same cell exploited patient identity to solve the contrastive task at ≥95% accuracy, then collapsed to LB 0.59. Mitigations that worked: (a) MIL auxiliary loss with per-patient mean-logit BCE, (b) AdaBN to absorb the train-to-test distribution shift, (c) test-set stain normalization, (d) `patient_id = −1` sentinel for pseudo cells so MIL skips them.

### M4. Borrowed results from papers rarely transferred, so validate each one

Three attempts to transfer published results to my setup failed (v37 modality-specific FL aug, v38 early fusion, v42 cross-modal contrastive SSL). The common cause: my 1-channel grayscale at 128 native input has a different signal/noise profile than the source papers' multi-channel emission stacks or higher-resolution variants. The one cross-paper bet that did pay (Lee 2013 pseudo-labels and Hinton 2015 distillation in v44/v46) succeeded because both are dataset-agnostic mechanisms that don't depend on signal-modality specifics.

### M5. The final recipe was built directly out of earlier failures

The v46 recipe is a direct response to the diagnosed failures in earlier versions: stripped the v37/v43 FL-aug suspect, kept the within-recipe seed ensemble (M2), used `patient_id = −1` to keep MIL patient-grouped on real cells (M3), and combined Lee 2013 + Hinton 2015 (M4) on top of the v41 regularizer floor (M1). Each component traces to a specific failure analysis in the rows above.
