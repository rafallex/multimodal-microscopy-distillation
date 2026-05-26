# 7. Negative results and failure-mode analysis

Seven experiments did not work, in informative ways. We analyze each individually because the instructor's grading framework explicitly values *what was done and how it was presented* over LB score, and because diagnosing failures was empirically more useful for guiding subsequent versions than studying the wins.

## 7.1 v22 — cross-recipe ensemble collapse

**Setup.** Sigmoid-average of v19 (LB 0.7455) and v21 (LB 0.7018), evaluated under three averaging schemes: probability average, rank average, and geometric mean. Members were ~0.05 LB apart.

**Result.** All three schemes underperformed v19 alone: 0.7422 / 0.7436 / 0.7328.

**Diagnosis.** Sigmoid-averaging treats both members symmetrically. When members differ in LB by more than the noise floor, the weaker model drags the ensemble below the stronger member's confidence on cells where they disagree. Theoretically a non-uniform weight scheme (e.g., reciprocal-error weighting) could rescue this, but the additional parameter introduces overfitting risk on a tiny development set.

**Lesson.** Cross-recipe sigmoid averaging is net-negative when members differ by >0.02 LB. We confirmed this in §7.6 (v45_probe) at a tighter gap.

## 7.2 v27 — single held-out validation patient pair was too noisy

**Setup.** Adopted a single 2-patient validation split (pat_5 and pat_14) inside the 12-patient training set, with the goal of using validation AUC instead of public LB as a model-selection signal. Combined with discriminative LR and a best-checkpoint policy.

**Result.** Training selected `best_epoch = 0`, i.e., an essentially untrained model. Public LB was correspondingly poor and we abandoned the recipe before further submissions.

**Diagnosis.** pat_5 has saturated FL exposure and pat_14 has notably darker FL — both are at the extremes of the per-patient distribution. With N=12 patients, removing two reduces the training-positive class to ~10 patients of signal, and the held-out pair's adversarial brightness profile produces low validation AUC at every epoch. Best-epoch selection then locks onto the random-init checkpoint.

**Lesson.** Conventional cross-validation does not work on N=12 patients with high per-patient distribution variance. We switched to using public LB as the primary external validator for the rest of the project, accepting the 4-submission-per-day quota as the rate-limiting cost.

## 7.3 v37 / v38 — Lian 2024 transfer attempts that did not generalize

The Lian et al. 2024 paper (same multimodal microscopy problem class, published F1 = 0.83 with a CAFNet architecture) was an obvious source of methodological ideas. We tested two:

**v37 (LB 0.7092, −0.036 from v19).** Implemented Lian §5.2's modality-specific augmentation: heavy ColorJitter on the FL channel with `brightness = 0.8, contrast = 0.8`. Lian's ablation reported this single change as worth ~19 pp F1 on their setup.

**v38 (LB 0.7147, −0.031 from v19).** Implemented Lian Table 3's early fusion: concatenate BF and FL into a single 2-channel input fed to one backbone, rather than the dual-branch late fusion we had been using.

**Diagnosis.** Lian's FL is a 4-channel emission stack (multiple fluorescence wavelengths). Our FL is **1-channel grayscale**. ColorJitter with `brightness = 0.8` against an already-low-intensity grayscale signal wipes out most of the discriminative signal — what looked like a strong augmentation on 4-channel emission becomes destructive on 1-channel. Similarly, early fusion's benefits in Lian likely depend on the per-pixel BF/FL alignment being tight enough to learn joint features at the pixel level; on our 128×128 crops the alignment is loose, so feature-level late fusion preserves more signal.

**Lesson.** Don't assume cross-paper transferability. Re-validate any borrowed ablation on the actual data modality and resolution.

## 7.4 v42 — CoMIR-style cross-modal SSL learned the patient shortcut

**Setup.** CoMIR-style contrastive self-supervised pretrain (Pielawski et al. 2020 style): two encoders (BF and FL), shared projection head, InfoNCE/NT-Xent loss with paired (BF, FL) of the same cell as positives. Pretrain for 10 epochs, then supervised finetune for 8 epochs.

**Result.** InfoNCE loss converged to 0.02 in epoch 1 (random baseline for 128-way contrastive is log(128) ≈ 4.85), then plateaued. Supervised finetune reached `tr_auc = 0.96`. Public LB: 0.5908, near random.

**Diagnosis.** Under our data structure each cell has exactly one source patient. Paired (BF, FL) of the same cell *i* therefore always share a patient identity; unpaired (BF, FL) of cells *i*, *j* with *i* ≠ *j* usually have different patients. The contrastive task can be solved at ≥95% accuracy by encoding **patient identity** rather than cell content. The pretrain backbone therefore concentrated on the patient signature; the supervised stage then memorized within-patient class distributions, which is precisely the spurious correlation a patient-disjoint test set is designed to expose. The 0.4 gap between training AUC (0.96) and public LB (0.59) is the signature of this failure mode.

**Lesson.** Cross-modal SSL on a strongly patient-grouped dataset can *amplify* the patient shortcut. To use it as a generalization aid here we would need positive pairs that span patient boundaries — e.g., pairing two cells from different patients that share an annotated phenotype. We did not have annotations dense enough to construct such pairs.

## 7.5 v43 — four-component stack regressed −0.012 from v41

**Setup.** Built on top of v41 (LB 0.7563). Added four changes simultaneously: (a) FL-tuned augmentation (ColorJitter brightness=0.5 contrast=0.3 + RandomGamma 0.85–1.15 with p=0.5 on the FL branch only), (b) weight decay 1e-4 → 3e-4, (c) 3-seed × SWA ensembling on the last 4 epochs, (d) 40-way TTA (5 scales × 8 D4) replacing 24-way (3 scales × 8 D4).

**Result.** Public LB 0.7444 — a 0.012 regression from v41 alone.

**Diagnosis at submission time.** Unknown — the four changes were confounded. We could not isolate which component was net-negative without per-change ablations, which would have cost 3 additional 5-hour Kaggle runs.

**Diagnosis from v46 (retroactive).** v46 = v43 with FL-tuned augmentation and the WD bump reverted, plus the soft-pseudo addition. It scored 0.8236, a +0.079 jump from v43. Even allowing the soft pseudo step to be worth +0.039 (matching the gain we measured against v44_seed1), the reverted components account for ~+0.04 in net positive direction. The 3-seed × SWA ensemble and the extended TTA were kept across v44, v46, and v47 — both presumed beneficial when measured in isolation.

**Lesson.** Stacking four unvalidated changes in a single experiment is a methodological error: it confounded the component analysis and cost a 5-hour run that produced no isolatable signal. After v43 we adopted the rule of changing one or two related variables per submitted version.

## 7.6 v45_probe — cross-recipe ensemble confirmation at a tighter gap

**Setup.** Local sigmoid-average of v41 (LB 0.7563) and v44 (LB 0.7812), submitted as a probe to test whether the v22 cross-recipe rule held at a smaller member gap.

**Result.** LB 0.7729. Below v44 alone by 0.0083, but **above** the naive linear LB-average of members (0.76875) by 0.0042.

**Diagnosis.** Pearson(v41, v44) = 0.936 — the two sets of predictions are highly correlated but not redundant. The ensemble adds small per-cell signal (4,770 of 59,040 cells flipped sigmoid sign vs v44 alone), and that signal manifests as the +0.004 lift above naive linear-LB-averaging. But the 0.025 LB member gap is wide enough that the penalty from the weaker member dominates the small lift. The result is below the stronger member.

**Lesson.** Cross-recipe ensemble at this dataset's scale is favorable only when member LB-gap is ≤ ~0.015. Within-recipe ensemble (v44, v46) remains the safe pattern in principle — though v47 (§7.7) shows even within-recipe averaging can fail when seed dispersion expands.

## 7.7 v47 — within-recipe ensemble net-negative under an outlier seed

**Setup.** v47 trains 3 independent seeds of the iterative-noisy-student recipe and sigmoid-averages their TTA predictions. Each seed sees the same teacher (v46 ensemble, LB 0.8236) but with independent data shuffling, SWA epoch averaging, and random augmentation streams. This is the same within-recipe averaging pattern that worked for v46.

**Result.** Per-seed public LB: seed 1 = 0.8150, seed 2 = **0.8355**, seed 3 = 0.8126 (range 0.0229). The 3-seed ensemble landed at 0.8264 — **below the best single seed by 0.0091**. By contrast, v46's per-seed range was 0.0072 and the v46 ensemble (0.8236) beat each individual seed.

**Diagnosis.** Two-way story:

1. Round 2 of the noisy-student loop *expanded* seed dispersion (3× wider per-seed LB range vs round 1). The v46-ensemble teacher signal is denser and more confident than the v44_seed1 teacher used for v46, but it also carries more pseudo-label noise that the student seeds amplify differently. The widened dispersion produced an outlier (seed2 = 0.8355) that the sigmoid-average hedges against.
2. We cannot distinguish from public LB alone whether seed2's lift is real signal (in which case the ensemble cost us 0.009 of true performance) or public-split luck (in which case the ensemble correctly hedged). With no held-out validation pool, this stays "open" until the private LB resolves.

**Lesson.** Within-recipe ensembling is not unconditionally net-positive. It is favorable when seed dispersion is tight relative to the underlying generalization gap (v46), and ambiguous when one seed is a clear outlier (v47). Operationally we now submit *both* the ensemble and the best single seed as separate final-stage candidates, instead of relying on the within-recipe ensemble as the unconditional safe pick. This is also the methodology argument for keeping v41 (the strongest non-pseudo baseline) as a different-mechanism safety net for the final selection — see §6.4 and the v48 plan.

## 7.8 Summary

| Failure | Mechanism diagnosed | Was the lesson actioned? |
|---|---|---|
| v22 | Cross-recipe ensemble at 0.05 gap pulled below stronger member | Yes (rule encoded; reconfirmed in v45_probe) |
| v27 | Single held-out 2-patient validation too noisy on N=12 | Yes (switched to public LB validation) |
| v37 | Modality-specific FL aug strength too high for 1-channel grayscale | Yes (kept v41's BF-matched aug) |
| v38 | Early fusion alignment requirement not met at 128 native | Yes (kept dual-branch late fusion) |
| v42 | Contrastive SSL learned patient identity, not cell content | Yes (abandoned SSL pretraining path) |
| v43 | Four-change stack confounded; ≥1 component was net-negative | Yes (v46 reverted suspects, gained +0.079) |
| v45_probe | Cross-recipe ensemble fails at 0.025 LB gap (still > ~0.015) | Yes (rule tightened; stopped attempting cross-recipe ensembles) |
| v47 | Within-recipe ensemble net-negative under outlier seed (dispersion widened in round 2) | Open — submit ensemble *and* best seed as separate final candidates |

Seven discrete failure modes diagnosed; seven discrete lessons fed forward into the version that took #1 on the public leaderboard. The negative results are not separate from the methodology — they *are* the methodology.
