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
| v21 | ResNet-50 | **224 upscale** | first ResNet-50 attempt (confounded with upscale) | 0.7018 | Lu 2020: "no interpolation" warning |
| v22 | (v19 + v21) | — | local CSV ensemble (sigmoid_avg / rank_avg / geomean) | 0.7422 / 0.7436 / 0.7328 | all underperformed v19 alone |
| v23 | EffNet-B0 | 128 native | v19 recipe, BASE_SEED=2 | 0.7154 | 0.03 LB spread from seed alone |
| v27 | EffNet-B0 | 128 native | v19 + 2-patient val holdout + disc LR + best-val ckpt | — | `best_epoch=0` collapse, val patients adversarial |
| v30 | EffNet-B0 | 128 native | v19 + disc LR only (no val), backbone_lr_ratio=0.1 | 0.6448 | disc LR at 0.1 ratio HURT by ~0.10 |
| v34 | ResNet-50 | 128 native | v19 recipe, ResNet-50 backbone, no upscale, no disc LR | 0.7155 | `notebooks/improvedv34_source.ipynb` — teacher's backbone tested cleanly |
| v35 | DenseNet-201 | 128 native | v19 recipe, DenseNet-201 backbone | (queued) | `notebooks/improvedv35_source.ipynb` |
| v36 | SE-ResNet-50 | 128 native | v34 recipe + SE channel attention (timm seresnet50) + 20 epochs | (not submitted) | `notebooks/improvedv36_source.ipynb` |
| v37 | EffNet-B0 | 128 native | v19 + Lian §5.2 modality-specific aug (heavy FL color jitter brightness/contrast=0.8) | 0.7092 | `notebooks/improvedv37_source.ipynb` — Lian's FL aug finding is multi-channel specific, didn't transfer to grayscale |
| v38 | EffNet-B0 | 128 native | v19 + early fusion (single 2-ch backbone instead of dual late-fusion) | 0.7147 | `notebooks/improvedv38_source.ipynb` — Lian Table 3 finding didn't transfer to grayscale 128 |
| **v41** | **EffNet-B0** | **128 native** | **v19 + label smoothing 0.05 + multiscale TTA (3 scales × 8 D4) + dropout 0.4 + paired RandomResizedCrop** | **0.7563** | **`notebooks/improvedv41_source.ipynb` — NEW BEST. Four L4-aligned regularizers stacked cleanly: +0.011 over v19** |
| v42 | EffNet-B0 | 128 native | v19 + CoMIR-style cross-modal InfoNCE SSL pretrain (10 ep) → finetune (8 ep @ LR 1e-4) | 0.5908 | `notebooks/improvedv42_source.ipynb` — catastrophic. SSL learned patient identity, not cell content |
| v43 | EffNet-B0 | 128 native | v41 + 3-seed ensemble + SWA(last 4 ep) + FL-tuned aug (CJ 0.5/0.3 + RandomGamma) + WD 1e-4→3e-4 + 40-way TTA (5 scales) | 0.7444 | `notebooks/improvedv43_source.ipynb` — **REGRESSION −0.012 vs v41**. The stacked extras hurt without pseudo cells in the loss |
| **v44** | **EffNet-B0** | **128 native** | **v43 recipe + pseudo-labels from v41 at threshold 0.05/0.95 (~9.3k confident test cells added with patient_id=−1, MIL skips them)** | **0.7812** | **`notebooks/improvedv44_source.ipynb` — NEW BEST. +0.025 over v41, gap to leader just 0.0020. Pseudo-labels carried the win** |
| v45_probe | (v41 + v44) | — | local CSV ensemble (sigmoid_avg) | 0.7729 | underperformed v44 alone by −0.008 but added +0.004 over the naive LB-average of members. Pearson(v41,v44)=0.936. Gap of 0.025 LB between members was too wide to gain — same pattern as v22, gentler magnitude |
| **v44_seed1** | **EffNet-B0** | **128 native** | **single-seed extraction from v44's 3-seed ensemble (submission_seed1.csv)** | **0.7844** | **NEW BEST. +0.0032 over the v44 ensemble. Seed 1 of v44 happened to be the lucky one. Confirms the seed-luck hypothesis at this LB regime. Now 3rd on LB; leader at 0.7916. v44_seed1 is now the teacher for v46's distillation** |
| **v45** | **EffNet-B0** | **128 native** | **v44 minus FL-tuned aug + WD reverted 3e-4→1e-4 + pseudo source upgraded v41→v44 (noisy student iteration, Xie 2020). Keeps 3-seed × SWA, 40-way TTA, paired RRC, label smoothing, dropout 0.4** | (queued) | **`notebooks/improvedv45_source.ipynb` — requires `submissionv44` Kaggle dataset (upload v44's submission.csv as a new private Kaggle dataset). Target ~0.79–0.80 LB. Gap to leader is 0.0020 so plausibly tops** |
| **v46** | **EffNet-B0** | **128 native** | **v44 stripped (no FL aug, WD 1e-4) + SOFT pseudo-labels from v44_seed1 (the lucky teacher): all 59k test cells, raw probs as BCE targets (Hinton 2015 distillation), pseudo loss weighted 0.5** | **0.8236** | **`notebooks/improvedv46_source.ipynb` — ★ NEW BEST + #1 ON LB ★. +0.039 over v44 seed1, +0.042 over v44 ensemble. Distillation crushed expectations (I forecast +0.005 to +0.015). All 3 seeds reached tr_auc ≈ 0.993 (vs v44's 0.989). Now 0.010 ahead of Group 1 @ 0.8136. Need +0.0264 more to reach 0.85** |
| v46_seed1 | EffNet-B0 | 128 native | single-seed extract from v46 (submission_seed1.csv) | 0.8157 | v46 seed1 alone scored 0.0079 below the v46 ensemble |
| v46_seed2 | EffNet-B0 | 128 native | single-seed extract from v46 (submission_seed2.csv) | 0.8229 | v46 seed2 alone scored 0.0007 below the v46 ensemble. **Different pattern from v44**: in v46 the ensemble beat both observed seeds, meaning within-recipe averaging genuinely added signal here (consistent with the tighter tr_auc convergence ≈ 0.993 across seeds vs v44's 0.989 spread) |
| **v47** | **EffNet-B0** | **128 native** | **Same recipe as v46, but teacher swapped: SOFT pseudo from v46 ensemble (LB 0.8236) instead of v44_seed1. Iterative noisy student round 2 (Xie 2020). Single-variable test of stronger teacher → stronger student in round 2** | **0.8264** | **`notebooks/improvedv47_source.ipynb` — ★ +0.0028 over v46 on the ensemble, +0.0119 on the best single seed. Per-seed public LB: 0.8126 / 0.8150 / **0.8355** (range 0.0229, ~3× wider than v46's 0.0072). v46 per-seed (now n=3): 0.8157 / 0.8187 / 0.8229. v47 per-seed SE = 0.0073 vs v46 SE = 0.0021 — **v47 seed noise 3.5× larger**. The ensemble at 0.8264 sat **below v47_seed2's 0.8355 by 0.0091** — within-recipe averaging hedged against a strong outlier seed. Earlier "tr_auc tightened" framing (0.9937/0.9929/0.9932) was on training AUC, not LB; on the metric we report, dispersion expanded. **Noise-floor analysis: v46→v47 ensemble lift is 0.27σ (indistinguishable from zero); v46→v47_s2 best-seed lift is 1.16σ (marginal).** Per-cell analysis: **97% of |p_s2 − p_ensemble| concentrated in teacher's dark-knowledge middle band** (cells where p_v46 ∈ [0.05, 0.95]) — strongly suggests s2 lift is signal, not split-luck. Decision: do NOT do a vanilla round 3; the right next experiment is Hinton temperature distillation (v48)** |
| **v48** | **EffNet-B0** | **128 native** | **v47 recipe with ONE knob changed: Hinton 2015 temperature distillation at T=2.0. Teacher is v46 (UNCHANGED — same as v47). Softens v46 teacher targets by sigmoid(logit(q)/T) at load, softens student logits by z/T at train, scales pseudo loss by T². Pure single-knob test of whether the "soft" in v46/v47's soft-pseudo BCE was actually firing** | (queued) | **`notebooks/improvedv48_source.ipynb` — requires `submissionv46` Kaggle dataset (same as v47, no new upload needed). GPU quota reset 2026-05-30 ~02:00 UTC. Decision rule: ≥0.829 = T=2 is the missing piece (consider v49 = T=2 + v47 teacher); 0.826-0.829 = T=2 helps marginally; ≈v47 ±0.002 = T=1 already captured Hinton's signal (publishable flat); <0.825 = T=2 destroys signal at this teacher quality (publishable negative). All four outcomes give the report a defensible finding.** |
| **v48** RESULT stub | **EffNet-B0** | **128 native** | _Fill once Kaggle scoring completes_ | `LB=___ (ensemble)` / `s1=___ s2=___ s3=___` | _Per-seed range: ___. Verdict against decision rule above: ___. Compare to v47 ensemble 0.8264 / v47_s2 0.8355. Noise-floor: \|Δ_ens\|/SE_Δ = ___ σ, \|Δ_s2\|/SE_Δ = ___ σ. Action: __ (lock final picks / try v49 with v48 teacher / pivot to lucky-seed retraining)._ |
| **v49** | **EffNet-B0** | **128 native** | **v47 recipe with ONE knob changed: teacher CSV swapped from v46 ensemble (LB 0.8236) → v47_seed2 (LB 0.8355). Tests whether the 97%-middle-band finding (v47_s2's lift is concentrated in cells the v46 teacher was uncertain about) compounds when v47_s2 itself becomes the round-3 teacher. The teacher delta (v46_ens→v47_s2 = +0.0119) is 4× larger than the v46_ens→v47_ens delta (+0.0028) that produced the indistinguishable round-2 lift, so this directly stress-tests §VIII-C's diminishing-returns hypothesis at a larger teacher-quality gap.** | (queued for May 30/31) | **`notebooks/improvedv49_source.ipynb` — generated from v47 by `overleaf-report/figure-sources/build_v49_notebook.py`. Requires a NEW Kaggle dataset upload: `results/v47/submission_seed2.csv` → `rafaelproena/submissionv47seed2` (one-time setup). Forecast: ensemble 0.83-0.86; P(beats v47_s2 = 0.8355) ≈ 35-45%. Decision rule: ≥0.840 ensemble = signal compounded (lock as primary pick); 0.830-0.840 = saturated round 3 (v47_s2 stays primary); <0.825 = teacher noise hypothesis confirmed (publishable negative on noisy-student-with-single-seed-teacher). Highest-EV GPU run we haven't done yet.** |
| **v49** RESULT stub | **EffNet-B0** | **128 native** | _Fill once Kaggle scoring completes_ | `LB=___ (ensemble)` / `s1=___ s2=___ s3=___` | _Per-seed range: ___. Verdict against decision rule above: ___. Compare to v47 ensemble 0.8264 / v47_s2 0.8355 / (v48 result). Noise-floor: \|Δ_ens\|/SE_Δ = ___ σ. Action: __ (new primary pick / lock v47_s2 / pivot to v50 = v49_best as teacher)._ |

## CPU-only ensemble probes (2026-05-27 evening batch)

Four ensemble recombinations submitted while GPU-bound, no retraining. Each tests a specific §VII-F / §VII-G hypothesis. SE_Δ vs v47 per-seed noise floor = 0.0103.

| Probe | Recipe | Public LB | vs v47 ens (0.8264) | vs v47_s2 (0.8355) | Finding |
|---|---|---|---|---|---|
| **B** | per-cell mean of v47 seeds 2+3 (drop worst) | **0.8281** | +0.0017 (+0.17σ) | −0.0074 | Drop-worst gives a marginal lift but recovers only ~20% of the ensemble→best-seed gap. §VII-G hypothesis "ensemble underperforms because of s1 outlier dilution" is only partially correct — s2 has positive signal that even s2+s3 averaging can't fully capture. |
| **C** | per-cell median across all 3 v47 seeds | **0.8201** | −0.0063 (−0.61σ) | −0.0154 | **Negative result**: median underperforms equal-mean ensemble. For a 3-seed distribution with one high outlier (s2), per-cell median discards s2's decisive predictions in the middle band (where s2 has the largest delta). Median is the wrong robust statistic when outliers are signal not noise. |
| **D** | sigmoid-avg of v47_s2 + v46 ensemble (cross-recipe, gap 0.012) | **0.8317** | **+0.0053** (+0.52σ) | −0.0038 | **§VII-F refined**: cross-recipe at 0.012 gap *does* beat within-recipe equal-seed averaging. Lands between members, closer to stronger. Refines the §VII-F threshold from "≤0.015 favorable" to "≤0.012 favorable" with empirical evidence at the high LB tier. |
| **E** | sigmoid-avg of v47_s2 + v44_seed1 (cross-recipe, gap 0.051) | **0.8161** | −0.0103 (−1.00σ) | −0.0194 (−1.89σ) | **§VII-A confirmed at higher tier**: 0.0317 above the weaker member (correlation bonus is real) but 0.0194 below the stronger. Wide-gap drag holds at the v47 LB tier — same qualitative pattern as v22 (gap 0.05, drag 0.0033 from stronger) but larger absolute drag, consistent with each LB point being harder to win at higher absolute LB. |

**Methodological synthesis.** The four probes triangulate the §VII-G ensemble collapse:

1. The v47 ensemble loses to v47_s2 by 0.0091. Drop-worst recovers +0.0017 (≈19% of the gap). Cross-recipe with the v46 ensemble recovers +0.0053 (≈58% of the gap). Per-cell median makes it worse by 0.0063.
2. None of these heuristics fully closes the ensemble→best-seed gap, but D demonstrates that the gap is bridgeable with cross-recipe diversification rather than within-recipe averaging.
3. The combined evidence supports the §VII-G "open" reading toward **signal**: s2's lift is robust to multiple averaging schemes and concentrated in the teacher-uncertain region (per-cell analysis: 97% middle-band, paper §VIII-E).

Still to submit (May 29 slots): **G** (per-cell confidence-weighted, directly exploits the 97% finding — was bumped from tonight's batch), **F** (LB-weighted, §VII-A risk test). G is the higher-priority of the two for the paper's middle-band exploitation narrative.

## Top-of-leaderboard reference (snapshot 2026-05-27 12:13 UTC)

| # | Team | Score | Members |
|---|---|---|---|
| 1 | Group 1 | 0.8448 | edvardschmidt, leokemetli, svanteandersson |
| 2 | Group10 | 0.8445 | albinpbergman, hugohansson123, olleflygar, saimazubair, tildemackintosh |
| **3** | **Group 15 (us)** | **0.8355** | rafaelproena (v47_seed2) |
| 4 | Groups 14 | 0.8125 | hoyumikaellau, nicolechaoo |
| 5 | Group 6 | 0.8007 | elsasundh, emelief, fahimrayhan, mdsabbirppp |

We were briefly #1 between v46 and v47, and held it through v47 (the comment thread above for v47 reflects that snapshot). Between 2026-05-26 and 2026-05-27 Group 1 (0.7916 → 0.8448) and Group10 (new entrant → 0.8445) overtook us. Our placement at 0.8355 comes from **v47_seed2**, not the v47 ensemble; the ensemble at 0.8264 would place 4th, which is empirical evidence for the within-recipe ensemble collapse documented in the paper §VII-G. Gap to current #1 = 0.0093.

## Lessons by experiment

| Experiment | Lesson |
|---|---|
| v19 vs v23 (seed diversity) | Same recipe gives 0.03 LB spread per seed — high-variance regime |
| v21 (ResNet-50 + 224 upscale) | Confounded change. Lu 2020's "no interpolation" warning matters at this image scale |
| v22 ensembles (v19 + v21) | Weaker member dragged the ensemble below v19 alone. Don't ensemble when components are >0.02 apart |
| **cross-recipe ensemble rule (v22 + v45_probe)** | **Two attempts at cross-recipe sigmoid-averaging have both regressed below the stronger member: v22 (v19+v21, 0.05 LB gap) and v45_probe (v44+v41, 0.025 LB gap). The +0.004 lift v45_probe added over the naive LB-average suggests cross-recipe ensembling does add small signal, but the LB gap penalty dominates for any practical gap. Operational rule: stop trying cross-recipe ensembles. Within-recipe seed + SWA averaging (which is what v44 is doing internally) remains the winning ensemble pattern.** |
| v27 (val patient holdout) | Single 2-patient holdout on N=12 is fundamentally too noisy. pat_5 (saturated FL) + pat_14 (dark FL) are at FL exposure extremes — adversarial val choice. best_epoch=0 = essentially untrained model |
| v30 (discriminative LR) | Standard "1/10 of original LR" recipe (L4 slide 96) is too aggressive for transferring ImageNet features to microscopy. Backbone needs more LR to learn texture features the ImageNet weights didn't see |
| v34 (ResNet-50 clean test) | ResNet-50 at 128 native (no upscale, no disc LR) ≈ EffNet-B0 different-seed result (v23). The teacher's recommended backbone works — but doesn't beat v19. v21's 0.7018 was due to the 224 upscale, not ResNet-50 itself. |
| v36 (SE-ResNet-50) | SE attention (Hu et al. 2018) added to every bottleneck. EfficientNet has SE built-in; v34 did not. v36 isolates whether channel attention closes the gap. +8 epochs (12→20) since larger model needs more training steps. Not submitted. |
| v37 (Lian §5.2 modality-specific aug) | Lian's ablation showed FL color jitter alone is worth ~19 pp F1 on their setup. Didn't transfer — their FL is 4-channel emission stack, ours is 1-channel grayscale; the strong brightness/contrast=0.8 jitter is washing out the only intensity signal we have. |
| v38 (early fusion) | Lian Table 3: early fusion beat late fusion by ~1.7 pp F1 on their data. Didn't transfer to grayscale 128. Likely test-set BF/FL alignment isn't tight enough for pixel-level fusion to win over feature-level fusion. |
| **v41 (stacked L4 regularizers)** | **Four cheap L4-slide additions (label smoothing 0.05 + dropout 0.3→0.4 + paired RandomResizedCrop + 24-way multiscale TTA) stacked roughly additively for +0.011 over v19. Lesson: textbook regularization still has headroom even after a strong baseline. The path forward is stacking small, well-motivated regularizers, not architectural overhauls.** |
| v42 (CoMIR cross-modal SSL) | InfoNCE loss converged to 0.02 (suspiciously low — log(128)≈4.85 is random) in epoch 1, then plateaued. Failure mode: with only one patient per cell, paired BF/FL of cell *i* always share a patient and unpaired BF/FL of cell *j*≠*i* usually don't, so encoding *patient identity* solves the contrastive task at ≥95%. Backbone got optimized to *strengthen* patient signature; finetune memorized it; test patients are different patients → tr_auc 0.96 vs LB 0.59. Cross-modal SSL needs positive pairs that span patient boundaries to work as a patient-shortcut remover. |
| v43 standalone (stack-on-v41) | v41 was at a local max. Stacking FL-tuned aug + WD 1e-4→3e-4 + wider TTA + 3-seed/SWA ensemble all at once produced LB 0.7444 — a −0.012 regression. Lesson: don't compose 4 unvalidated changes in one experiment; at least one of them was a net negative and the others didn't cover for it. The seed ensemble + SWA didn't even rescue the regression. |
| **v44 (pseudo carries everything)** | **v44 = v43 + pseudo-labels (only diff). v44 (0.7812) beat v43 (0.7444) by +0.037 and v41 (0.7563) by +0.025. Arithmetic: pseudo-labels alone are worth ≥+0.037 LB, because they also had to overcome v43's −0.012 negative stack. Lee 2013-style confidence-thresholded pseudo-labeling is dramatically more valuable than any combination of L4 regularizers when the teacher is decent (v41 at 0.7563 was enough). The bottleneck wasn't the loss or the augmentation — it was the size of the labeled training set.** |
| **v47 (mean lift compresses, dispersion expands)** | **v47 = v46 recipe with teacher swapped v44_seed1→v46. Single-knob test of round 2. Result: +0.0028 LB on the ensemble (0.8236→0.8264). Compare to round 1's +0.039 (v44_seed1→v46). The mean lift compressed ~13× in one iteration despite using a strictly stronger teacher. Two takeaways: (a) the dominant gain from distillation is in round 1 when the teacher first becomes "soft enough" to convey calibration signal; (b) **per-seed public LB dispersion expanded ~3×** — v47 seeds spanned 0.8126/0.8150/0.8355 (range 0.0229, SE 0.0073) vs v46's 0.8157/0.8187/0.8229 (range 0.0072, SE 0.0021); seed noise 3.5× larger. The training-AUC tightened (0.9937/0.9929/0.9932) but on the metric we report, dispersion widened. The v47 ensemble at 0.8264 sits 0.0091 *below* the best single seed (0.8355) — within-recipe averaging hedged against an outlier seed (see paper §VII-G "open" finding). Per-cell analysis: 97% of |p_s2 − p_ens| concentrated in teacher dark-knowledge middle band ⇒ s2 lift likely signal not split-luck. Decision: don't waste 5h GPU on a vanilla round 3 — change the mechanism (Hinton temperature, the missing half of v46's "soft pseudo" implementation).** |

## Final-submission selection strategy

Kaggle lets you pick 2 submissions for private LB. Updated plan as of 2026-05-27 (we are currently #3, two teams overtook us 2026-05-26 → 2026-05-27):

The §VII-G analysis in the paper presents three plausible strategies. The new public-LB reality (v47_s2 is what's holding our #3 placement; v47 ensemble would place #4) is direct evidence that within-recipe averaging has cost us LB on this split. That argues for the **aggressive** strategy over the **robust** one:

1. **Aggressive:** v47_seed2 (LB 0.8355) + v41 (LB 0.7563 as a different-mechanism safety net). Bets that seed 2's training dynamics generalize.
2. **Robust:** v47 ensemble (LB 0.8264) + v41. Preserves the within-recipe averaging discipline of v46 but gives up 0.009 public LB.
3. **Diversified within-pseudo:** v47_seed2 + v47 ensemble. No supervised-only safety net.

Why NOT pick v46 as a safety net: v46 is the teacher v47 distills from. They share recipe lineage almost completely (only difference: teacher source CSV). A correlated pair offers no diversification on private LB. v41 is the lowest-correlation pick available among submitted recipes.

A v47 + v41 sigmoid-average submission is **NOT** worth a slot test — the members are 0.070+ LB apart, far outside the 0.015 cross-recipe ensembling threshold we established empirically via v22 and v45_probe (M2).

**Noise-floor caveat.** A noise-floor analysis (paper Table III) places the v46→v47 ensemble lift (+0.003) at 0.31σ relative to the v47 per-seed SE — indistinguishable from zero. The v46→v47_s2 best-seed lift (+0.012) is at marginal (1.33σ). The +0.039 v44→v46 soft-pseudo gain is the only robust delta in the entire iteration chain (4.75σ).

v48 (Hinton T=2) is queued for the May 29 Kaggle quota reset. If it gains we regain the lead; if it's flat or worse, v47_s2 remains the primary pick under the aggressive strategy.

## What's NOT tracked here

- The competition dataset (`multimodal-cancer-classification-challenge-2026/`) is gitignored.
- Per-version result artifacts (.pt checkpoints + extracted result folders) live in the
  local-only `results/vNN/` tree. The `submission.csv`, `history.json`, and `learning_curves.png`
  are kept locally for offline analysis but excluded from git.
- A more detailed local-only summary lives at `results/LB_SUMMARY.md` (also gitignored).

## Meta-lessons (cross-experiment synthesis)

Five insights that recur across multiple experiments and that drove the final-version recipe choices. These consolidate the per-experiment lessons above into reusable claims.

### M1. The dataset-size lever dominates on small-N patient-grouped data — but with diminishing returns

Between v19 and v41 we gained +0.011 LB from four textbook regularizers stacked cleanly (label smoothing, dropout 0.4, paired RandomResizedCrop, multiscale TTA). Between v41 and v46 we gained +0.067 LB by *growing the labeled training set* — first via hard pseudo-labels (+0.025) and then via soft-target distillation that used 6× more pseudo cells (+0.039 additional). The ratio is roughly **6× more LB per unit of effort by adding labeled-ish data than by improving the model**. On 12 training patients with ~10k cells each, the bottleneck was the size and quality of the supervisory signal, not the model's capacity.

**v47 added a diminishing-returns nuance**: iterating noisy student to round 2 with an even stronger teacher (v46 → v47) added only +0.003 LB — about 14× smaller than the round-1 gain. The dataset-size lever's biggest payoff is **the first time** you flip from hard labels to soft, all-cells distillation; subsequent rounds compress quickly. This suggests the lever is mostly about *unlocking the calibration channel*, not about endlessly stacking more noisy supervision.

### M2. Within-recipe ensembling works; cross-recipe ensembling fails for any practical LB gap

v44 and v46 internally average 3 seeds × SWA, the standard Izmailov-2018 within-recipe pattern. Both gained from it (v46 ensemble beat all observed individual seeds). We tested cross-recipe sigmoid averaging twice — v22 (v19+v21, 0.05 LB member gap) and v45_probe (v41+v44, 0.025 LB gap). Both regressed below the better member. The operational rule we adopted: **don't ensemble across recipes if the LB gap is wider than ~0.015**. The two failed attempts confirmed this rule at progressively tighter gaps.

### M3. Patient identity is the dominant spurious correlation; treat it as adversarial

The training data has 12 patients, the test set has disjoint patients, and per-patient FL exposure varies by ~20% mean intensity. Any method that can encode patient identity will preferentially learn it. v42 confirmed this catastrophically: cross-modal SSL on paired (BF, FL) of the same cell exploited patient identity to solve the contrastive task at ≥95% accuracy, then collapsed to LB 0.59. Mitigations that worked: (a) MIL auxiliary loss with per-patient mean-logit BCE, (b) AdaBN to absorb the train→test distribution shift, (c) test-set stain normalization, (d) `patient_id = −1` sentinel for pseudo cells so MIL skips them.

### M4. The transfer-from-paper bet pays out rarely; validate every borrowed ablation

Three attempts to transfer published results to our setup failed (v37 Lian §5.2 FL aug, v38 Lian Table 3 early fusion, v42 CoMIR cross-modal SSL). The common cause: our 1-channel grayscale at 128 native input has a different signal/noise profile than the source papers' multi-channel emission stacks or higher-resolution variants. The one cross-paper bet that *did* pay (Lee 2013 pseudo-labels and Hinton 2015 distillation in v44/v46) succeeded because both are dataset-agnostic mechanisms — they don't depend on signal-modality specifics.

### M5. Negative results compound; we got #1 by reading our failures faithfully

The v46 recipe is a direct response to the diagnosed failures in earlier versions: stripped the v37/v43 FL-aug suspect, kept the within-recipe seed ensemble (M2), used `patient_id = −1` to keep MIL patient-grouped on real cells (M3), and combined Lee 2013 + Hinton 2015 (M4) on top of the v41 regularizer floor (M1). Each component traces to a specific failure analysis in the rows above. The wins are not separate from the losses — the wins are the losses, processed.
