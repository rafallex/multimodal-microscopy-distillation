# 6. Results

We ran 29 logged Kaggle iterations between v15 (early baseline) and v46 (current best). The complete record with per-version recipe diffs, public-LB scores, and one-line lessons is maintained in [`LB_HISTORY.md`](../LB_HISTORY.md). This section summarizes the headline progression, the four most informative ablations, and the within-recipe ensembling result.

## 6.1 Headline progression

Figure 1 plots the chronological public-LB sequence with the most consequential inflections annotated.

![Figure 1: Public LB progression across 29 logged iterations.](figures/lb_progression.png)

The arc is:

- **v19 (LB 0.7455)** — baseline EfficientNet-B0 dual-branch with MIL aux loss, AdaBN, test-stain normalization, and strong augmentation. This was our reference floor for ~3 weeks.
- **v41 (LB 0.7563, +0.011)** — stacked four textbook regularizers on top of v19: label smoothing ε=0.05, dropout 0.4 (up from 0.3), paired RandomResizedCrop, and 24-way multiscale TTA. A clean L4-aligned win, additive across components.
- **v43 (LB 0.7444, −0.012 regression)** — added four further changes simultaneously (FL-tuned augmentation, WD 1e-4 → 3e-4, 3-seed × SWA ensemble, 40-way TTA). The stack regressed; see §7.5 for the post-mortem.
- **v44 (LB 0.7812, +0.025 over v41)** — added Lee 2013–style hard pseudo-labels from v41 (threshold 0.05 / 0.95, ≈9,350 confident test cells with `patient_id = −1` so the MIL loss skips them). The single most impactful change of the project until v46.
- **v46 (LB 0.8236, +0.039 over v44_seed1)** — soft-target distillation (Hinton 2015): instead of hard-thresholded pseudos, we use the teacher's raw probabilities as BCE targets for all 59,040 test cells, weighted at 0.5. Took #1 on the public leaderboard.

The headline number is a **+0.078 LB lift over the v19 baseline**, with the soft-pseudo step (v44 → v46) contributing exactly half of that on its own.

## 6.2 Version progression table

| Version | Recipe summary | Public LB | Δ |
|---|---|---|---|
| v15 / v16 | ResNet-18 baselines (early MIL + LOPO CV) | — | — |
| v19 | EffNet-B0 + MIL + strong aug + AdaBN + test stain norm + 8-way D4 TTA | **0.7455** | reference |
| v20 | new aug recipe (failed) | 0.5974 | abandoned |
| v21 | ResNet-50 + 224× upscale (confounded change) | 0.7018 | −0.044 |
| v23 | v19 recipe, different seed | 0.7154 | −0.030 (seed variance) |
| v27 | v19 + 2-patient validation holdout (failed) | — | `best_epoch=0` collapse |
| v30 | v19 + discriminative LR 1:10 | 0.6448 | −0.101 |
| v34 | ResNet-50 at 128 native (no upscale, no disc LR) | 0.7155 | clean backbone test |
| v37 | v19 + Lian §5.2 modality-specific FL aug | 0.7092 | −0.036 |
| v38 | v19 + early fusion (single 2-ch backbone) | 0.7147 | −0.031 |
| **v41** | v19 + label smoothing 0.05 + dropout 0.4 + paired RRC + multiscale TTA | **0.7563** | **+0.011** |
| v42 | v19 + CoMIR-style cross-modal InfoNCE SSL | 0.5908 | catastrophic (§7.4) |
| v43 | v41 + FL-tuned aug + WD bump + 3-seed × SWA + extended TTA | 0.7444 | **−0.012 regression** |
| **v44** | v43 + hard pseudo-labels from v41 (thr 0.05/0.95) | **0.7812** | **+0.025 over v41** |
| v44_seed1 | single-seed extract from v44 | 0.7844 | +0.003 (seed luck) |
| v45_probe | sigmoid average of v41 + v44 (local CSV ensemble) | 0.7729 | regressed below v44 alone |
| v46_seed1 | single-seed extract from v46 | 0.8157 | below v46 ensemble |
| v46_seed2 | single-seed extract from v46 | 0.8229 | below v46 ensemble |
| **v46** | v44 stripped + soft pseudo from v44_seed1 (all 59k cells, raw probs, weight 0.5) | **0.8236** | **+0.039, #1 on public LB** |
| v47 | v46 with v46 as new teacher (noisy-student round 2) | (queued) | — |

## 6.3 Key ablations and their lessons

### Backbone capacity at 128 native: not the bottleneck (v34 vs v19)

v34 (ResNet-50, same recipe as v19, no upscale, no discriminative LR) landed at 0.7155 — about the same as v23 (v19 with a different seed). The teacher-recommended backbone works, but at 128 native input it does not unlock additional capacity that v19's EfficientNet-B0 cannot already use. v21's earlier 0.7018 came from the 224× upscale (Lu 2020's "no interpolation" warning), not from ResNet-50 itself.

### Discriminative learning rate hurts in this regime (v30)

Applying the L4 slide's standard "1/10 backbone LR" recipe regressed by ≈0.10 LB. The ImageNet pretrain's texture features need more LR, not less, to adapt to grayscale microscopy. We kept a single global LR for all subsequent versions.

### Stacking unvalidated regularizers is dangerous (v43)

v43 added four changes simultaneously on top of v41 and lost 0.012 LB. The component test was confounded and we couldn't isolate the culprit until we ran v46 (which selectively reverted FL-tuned augmentation and the WD bump) and saw a +0.039 lift. By process of elimination, those two were the suspect contributors; the 3-seed × SWA ensemble and extended TTA were kept and presumed beneficial.

### Hard vs soft pseudo-labels: a 6× expansion of the labeled training set (v44 vs v46)

The hard-pseudo threshold of 0.05 / 0.95 in v44 used ~9,350 of 59,040 test cells (16%). v46's soft formulation uses **all 59,040**, weighting each cell's contribution by the teacher's confidence. The training set grew from 124k effective cells to 173k, and the soft probabilities preserve the "dark knowledge" (Hinton 2015) — the directional + magnitude information of teacher uncertainty that hard binarization discards. The +0.039 LB jump is the single largest gain we measured.

## 6.4 Within-recipe vs cross-recipe ensembling

v44 internally averages 3 seeds × SWA per the standard within-recipe pattern (Izmailov 2018). For v44 this gave a result (0.7812) that was actually below its lucky individual seed (0.7844). For v46 the opposite held: the 3-seed ensemble at 0.8236 beat all observed seeds (seed1 = 0.8157, seed2 = 0.8229, seed3 not submitted). The difference reflects how tightly the seeds converged — v46's per-seed `tr_auc ≈ 0.993` was a tighter band than v44's `0.989`, leaving more room for averaging to add signal.

Cross-recipe ensembling, by contrast, failed twice (v22 in §7.1, v45_probe in §7.6). The operational rule we adopted: keep within-recipe seed × SWA averaging on by default, do not attempt cross-recipe sigmoid-averaging unless the members are within ~0.02 LB of each other.
