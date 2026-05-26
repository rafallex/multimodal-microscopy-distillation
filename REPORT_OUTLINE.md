# Assignment 3 Report — Outline & Drafting Guide

> Working scaffold for the A3 report. Each section lists what to cover, key numbers from `LB_HISTORY.md`, and suggested figures/tables. Use it as a writing guide; elaborate in your own voice.

**Target length:** ~6–10 pages (typical for a project report at this level).
**Tone:** Methodological — the instructor explicitly grades *"what is done and how that is presented"*, not the LB score. Lead with your reasoning, not just your numbers.

---

## 1. Abstract (~150 words)

- Problem: binary cell-level cancer classification on multimodal microscopy (paired bright-field BF + fluorescence FL), 128×128, 12 training patients × ~10k cells/patient, ~38.8% positive.
- Approach: dual EfficientNet-B0 with late concat fusion + per-patient MIL aux loss, then progressive layering of L4-textbook regularizers, followed by a **pseudo-label pipeline** culminating in **Hinton-2015-style soft-target distillation** and **iterative noisy-student training**.
- Headline result: public LB **0.7455 → 0.8264** across 30 iterations, currently **#1 on the public leaderboard with a +0.013 lead over next-best**. The single largest gain (+0.039 LB) came from soft pseudo-labels (v46), not from architecture or regularization changes; a further +0.003 came from iterative noisy student round 2 (v47).
- Key methodological insight: **dataset size dominated architectural choices** on this problem. The bottleneck was labeled training set size, not model capacity or regularization.

---

## 2. Introduction (~0.5 page)

- The challenge: identify malignant cells in microscopy crops, paired BF + FL channels, given a strict patient-grouped split (test patients ≠ train patients).
- Why this is hard: only 12 training patients, large per-patient FL exposure differences, severe overfitting risk if patient identity leaks into the model.
- The two specific contributions of this work:
  1. **A semi-supervised pipeline that uses the test set itself** (pseudo-labels + distillation + iterative noisy student) to overcome the small-label-set bottleneck, lifting LB by **+0.070** over a strong supervised baseline (v41 → v47).
  2. **A systematic record of failures** (cross-recipe ensemble collapse, SSL patient-shortcut, regularizer stack regression) that clarifies *why* certain methods don't transfer to this setup.
- Roadmap for the rest of the paper.

---

## 3. Dataset & Problem Setup (~0.5 page)

- **Data summary:**
  - Train: 114,302 cells from 12 patients (pos rate 0.3876)
  - Test: 59,040 cells from disjoint patients (no overlap)
  - 128×128 grayscale BF + grayscale FL, paired by filename
- **Patient-grouped MIL nature:** each patient is a "bag"; cells within a bag share the patient-level diagnosis but can be heterogeneous. Standard cell-level BCE alone exploits within-patient correlation, so we add a per-patient mean-logit BCE auxiliary loss.
- **Test-time distribution shift:** measured FL test mean is 19.7% brighter than FL train mean. Motivates AdaBN and test-set stain normalization.
- **Class imbalance handling:** `pos_weight = neg/pos ≈ 1.58`, computed from real training cells only.
- **Suggested figure:** sample montage — 3 BF/FL pairs (one positive, one negative, one ambiguous) with patient ID labels.

---

## 4. Methods (~1.5 pages)

### 4.1. Backbone & Fusion Architecture

- Two independent EfficientNet-B0 branches (timm `efficientnet_b0.ra_in1k`, ImageNet pretrained, ~4M params each)
- 1-channel grayscale adapter: averaged the 3 RGB conv_stem weights into a 1-channel kernel to preserve as much pretrain signal as possible (vs random init)
- **Late concat fusion** after global average pooling → 2560-dim feature → 512-dim MLP head → 1 logit
- Total parameters: ~9.3M
- *Suggested figure:* architecture diagram with both branches, late concat, MLP head

### 4.2. Loss Functions

- **Cell-level BCE** with label smoothing ε=0.05 and `pos_weight`
- **Patient-level MIL aux loss** (weight 0.5): for each patient in the batch, mean the cell logits, then BCE against the patient label. Skipped for pseudo cells (patient_id = -1).
- **v46/v47 distillation loss** (per-cell):
  - real cell: BCE against smoothed hard label
  - pseudo cell: BCE against raw teacher probability, weight 0.5
  - implemented via `torch.where(real_mask, smoothed_real, pseudo_target)`

### 4.3. Training Recipe

- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- Schedule: OneCycleLR (pct_start=0.1)
- 12 epochs, batch size 128, gradient clipping 1.0
- **Patient-balanced sampler:** 4 patients × 32 cells per batch (with pseudo cells treated as a 13th "patient group")
- **SWA** over the last 4 epochs (Izmailov 2018), with a manual BN refresh pass for the averaged model (since standard `update_bn` doesn't support our (bf, fl) two-input signature)

### 4.4. Augmentation

- Paired geometric: D4 + paired ±10° rotation + paired affine (±15°, ±10% translate) + paired RandomResizedCrop (scale 0.85–1.0)
- Per-modality color: ColorJitter(0.4, 0.4) on both, RandomErasing p=0.25
- We **tested** stronger modality-specific FL aug (CJ 0.5/0.3 + RandomGamma) in v37/v43 — both regressed; v46/v47 use the v41 BF-matched aug

### 4.5. Adaptive Batch Normalization & Stain Normalization

- AdaBN (Li 2016): single forward pass over the test set in `train()` mode before inference, updating BN running stats to test-set statistics
- Test stain normalization: normalize all inputs using pixel mean/std measured on the test set rather than the train set (a 19.7% FL mean gap makes this measurable)

### 4.6. Test-Time Augmentation

- 40-way TTA: 5 input scales (96, 112, 128, 144, 160) × 8 D4 group elements
- Predictions averaged in probability space (post-sigmoid)
- 3-seed within-recipe ensembling on top (SWA average of each seed, then sigmoid-average across seeds)

### 4.7. Semi-Supervised Pipeline

- **Hard pseudo (v44, Lee 2013):**
  - Run a teacher model (v41) on the test set
  - Keep cells with prediction < 0.05 (confident negative) or > 0.95 (confident positive)
  - Assign hard 0/1 labels, patient_id = -1
  - Merge into the training set (~9,350 cells, +8% size)
- **Soft pseudo / distillation (v46, Hinton 2015):**
  - Keep **all** 59,040 test cells (6× more than hard pseudo)
  - Use the teacher's raw probabilities as BCE targets directly
  - Weight pseudo loss at 0.5 to leave real labels dominant
  - Combined training set: 173,342 cells (+52% over baseline)
- **Iterative noisy student (v47, Xie 2020):**
  - Round 1 student (v46) becomes round 2 teacher
  - Same distillation recipe, stronger soft targets

---

## 5. Experimental Protocol (~0.5 page)

- All experiments tracked in `LB_HISTORY.md` with exact recipe diff, public LB, and a one-sentence lesson.
- Public Kaggle LB used as the primary external validator (no held-out validation set; with only 12 patients a 2-patient holdout was tried in v27 and was too noisy — see §6.3).
- Compute: Kaggle Notebooks free tier (T4 ×2, ~5–6h per run with pseudo-labels). No HPC used.
- Submission discipline: 4 daily Kaggle slots, used carefully to test specific hypotheses rather than churn variants.

---

## 6. Results (~2 pages)

### 6.1. Version progression headline

- **Suggested figure (the report's centerpiece):** LB vs. version-index line chart, with annotations on key versions (v19 baseline, v41 regularizer stack, v44 hard pseudo, v46 soft pseudo, v47 noisy student round 2). The chart is built and current at `presentation/figures/lb_progression.png` (PNG for the PPT) and `overleaf-report/lb_progression.pdf` (PDF for the LaTeX paper).
- Headline: 0.7455 (v19) → 0.8264 (v47), a **+0.081 LB lift**

### 6.2. Component table

Suggested table format:

| Version | Recipe summary | LB | Δ vs prev |
|---|---|---|---|
| v15 | ResNet-18 baseline | — | — |
| v19 | EffNet-B0 + MIL + strong aug + AdaBN + test stain norm | 0.7455 | — |
| v34 | ResNet-50 same recipe | 0.7155 | −0.030 (backbone test) |
| v41 | v19 + label smoothing 0.05 + dropout 0.4 + paired RRC + 24-way TTA | 0.7563 | +0.011 |
| v43 | v41 + FL-tuned aug + WD 3e-4 + 3-seed × SWA + 40-way TTA | 0.7444 | **−0.012** ← regression |
| v44 | v43 + hard pseudo from v41 | 0.7812 | +0.025 over v41 |
| v44 seed1 | v44 single-seed extract | 0.7844 | +0.003 over ensemble |
| v46 | v44 stripped + soft pseudo from v44_seed1 | **0.8236** | **+0.039** ← biggest single gain |
| **v47** | **v46 with v46 ensemble as teacher (iterative noisy student round 2, Xie 2020)** | **0.8264** | **+0.003 ← still #1 (+0.013 over next-best)** |
| v48 | v47 with Hinton T=2 temperature distillation (single-knob T test, v46 teacher) | (queued, May 29) | — |

### 6.3. Key ablations

For each, explain the hypothesis and the result:

- **Backbone (v34 vs v19):** ResNet-50 at 128 native input ≈ EffNet-B0 different-seed result. Larger ImageNet-tier backbones don't transfer free benefit at this image scale.
- **Discriminative LR (v30):** standard "1/10 backbone LR" recipe (L4 slide) hurt by 0.10 LB. ImageNet→microscopy transfer needs the backbone to learn texture, not just freeze.
- **Stacking regularizers (v43):** 4 simultaneous changes (FL aug + WD bump + multi-seed + extended TTA) regressed by 0.012 against v41. Confounds, doesn't isolate the bad component.
- **Hard vs soft pseudo (v44 vs v46):** soft used 6× more cells (59k vs 9.3k) with richer per-cell signal → +0.042 lift over hard pseudo with the same teacher lineage.
- **Cross-recipe ensembling (v22, v45_probe):** two attempts, both regressed below the better member. Within-recipe ensembling (seed × SWA) worked; cross-recipe did not.

---

## 7. Negative Results and Failure-Mode Analysis (~1 page)

This is your highest-leverage section — analyses of why methods *didn't* work distinguish a methodology report from a leaderboard report.

### 7.1. v22 — Cross-Recipe Ensemble Collapse

- Sigmoid-averaged v19 + v21 (members ~0.05 LB apart). All three averaging methods (sigmoid_avg, rank_avg, geomean) underperformed v19 alone.
- **Lesson:** members must be within ~0.02 LB to ensemble net-positively. Confirmed later in v45_probe.

### 7.2. v27 — Single Held-Out Validation Failure

- Used pat_5 + pat_14 as held-out validation (one FL-saturated, one FL-dark). `best_epoch=0` collapse.
- **Lesson:** with N=12 patients and per-patient FL exposure heterogeneity, a 2-patient held-out is too noisy and adversarial. Public LB became the de-facto validation.

### 7.3. v37 / v38 — Failed Lian-Paper Transfer

- v37: copied Lian §5.2's heavy FL color jitter (brightness=0.8, contrast=0.8) → −0.036 LB regression
- v38: copied Lian Table 3's early fusion → −0.030 LB regression
- **Lesson:** the original Lian paper used 4-channel emission stacks at higher resolution; their findings don't transfer to 1-channel grayscale at 128. Augmentation strength and fusion strategy are dataset-specific.

### 7.4. v42 — CoMIR-Style Cross-Modal SSL Collapse

- InfoNCE pretrain on paired (BF, FL): contrastive loss converged to 0.02 in epoch 1 (random would be log(128)≈4.85), suggesting an easy non-task solution.
- Diagnosis: with one patient per cell, paired BF/FL of cell *i* always share a patient; unpaired BF/FL of cells *j*≠*i* usually don't. The model can solve the contrastive task at ≥95% by encoding **patient identity** rather than cell content.
- Finetuned model: tr_auc 0.96, public LB 0.5908 (random-baseline-tier).
- **Lesson:** SSL on a strongly patient-grouped dataset can amplify the patient shortcut. Cross-modal contrastive needs positive pairs that span patient boundaries to work as a generalization aid here.

### 7.5. v43 — Stacked Regularizer Regression

- v41 + (FL-tuned aug + WD 3e-4 + 3-seed × SWA + 40-way TTA): 4 simultaneous additions
- Result: −0.012 LB regression vs v41 alone
- **Lesson 1:** never stack 4 unvalidated changes in one experiment — can't isolate the culprit.
- **Lesson 2:** at least one of FL-tuned aug or WD bump was net-negative; v46 reverted both and kept the seed ensemble + extended TTA (likely beneficial in isolation).

### 7.6. v45_probe — Cross-Recipe Ensemble (Confirmation of §7.1)

- Sigmoid-average of v41 (LB 0.7563) + v44 (LB 0.7812) → 0.7729
- Underperformed v44 alone by 0.008, but +0.004 above the naive linear LB-average
- **Lesson:** confirmed cross-recipe ensemble rule at a smaller gap (0.025 LB). Cross-recipe averaging adds tiny signal but the LB-gap penalty dominates for practical gaps. Within-recipe ensemble (v44's 3 seeds × SWA) is the only ensembling mode that's reliably positive here.

---

## 8. Discussion (~1 page)

### 8.1. The dataset-size lever dominates

- Architecture and regularization between v19 and v41 gained +0.011 LB combined.
- The single change "add ~9k confident test cells with hard labels" gained +0.025 LB.
- The single change "use raw probabilities for all 59k test cells" gained +0.039 LB.
- **Interpretation:** with 12 patients and 114k labeled cells, the model is data-bound, not capacity-bound. Methods that effectively grow the labeled set dominate.

### 8.2. Why soft pseudo beat hard pseudo so dramatically

- Hard pseudo at threshold 0.05/0.95 discards 84% of test cells (49.7k of 59k)
- Soft pseudo keeps all 59k cells, with each cell's contribution weighted by the teacher's confidence
- Hinton's "dark knowledge": a cell labeled p=0.78 by the teacher carries directional + magnitude information that hard binarization throws away
- Empirically: soft beat hard by +0.042 LB on this dataset

### 8.3. Iterative noisy student (Xie 2020) — when does it pay off?

- Round 1 (v46) was a +0.039 lift over its teacher (v44_seed1).
- Round 2 (v47) tested whether the lift compounds with a stronger teacher (v46 ensemble). Result: **+0.003 LB** — about a 14× compression from round 1.
- **Per-seed `tr_auc` tightened** from v46's 0.989–0.993 spread to v47's 0.993–0.994 band. So even when round-2 contributes little to the *mean* it acts as a **variance reducer** — useful for private-LB stability under split shake-out.
- General principle re-stated: iterative noisy student's biggest payoff is **the first time** you flip from hard pseudo to soft-all-cells distillation. Subsequent rounds iterate the same channel with a marginally better-calibrated teacher and the marginal lift collapses quickly. To get further gains the *mechanism* (not the iteration count) needs to change — which is what v48 tests via Hinton temperature softening (v47 nominally uses "soft pseudo" but with T=1, so Hinton's actual softening prescription was never exercised).

### 8.4. What did NOT work and why

- Brief reframe of §7 negative results as a coherent argument:
  - Cross-recipe ensembling collapses when members differ in LB by more than ~0.02
  - SSL collapses when positive pairs encode patient identity rather than cell content
  - Standard L4 regularizers (discriminative LR, modality-specific aug) don't all transfer to this microscopy/grayscale/small-N setup

### 8.5. Limitations and threats to validity

- **Single public LB number per submission, no held-out set.** Some lifts may not survive private LB (test-set re-shuffling). Mitigation: kept v41 (the strongest non-pseudo baseline) as a private-LB safety pick.
- **Compute budget capped at 4 Kaggle submissions/day.** Limited the number of ablations runnable.
- **No instructor pre-approval of semi-supervised methods.** Mitigated by the May 14 announcement clarifying "all tools allowed" and explicitly endorsing pretrained models.
- **Possible overfitting to public LB.** Aggressive pseudo-label iteration risks tailoring to the public split. v47 is the test of whether the recipe still generalizes when the teacher gets stronger.

---

## 9. Final Submission Strategy (~0.25 page)

Kaggle private LB picks (2 max):

1. **v47 (LB 0.8264)** — primary. Iterative noisy student round 2 (v46 ensemble as teacher) + 3-seed × SWA + 40-way TTA. The methodology's strongest standalone result.
2. **v41 (LB 0.7563)** — safety net. No pseudo at all (different mechanism), so it hedges against any pseudo-label overfitting to the public split.

We deliberately avoid using v44 or v46 as the safety pick because both share the pseudo-label lineage with v47 (v46 IS v47's teacher). A correlated pair would offer no diversification.

If v48 (Hinton T=2) lands above v47, the primary slot upgrades to v48 and the v41 safety net stays unchanged.

---

## 10. Conclusion (~0.25 page)

- Summary of the journey: 30 iterations, +0.081 LB lift, #1 on public board with +0.013 lead
- Three takeaways:
  1. On small-N patient-grouped datasets, growing the *effective* labeled set via pseudo-labels and distillation outperforms architectural or regularization gains by an order of magnitude.
  2. Cross-recipe ensembling has a hard LB-gap limit (~0.02 in this regime).
  3. Negative results — SSL collapse, regularizer regression, ensemble failure — taught more about the dataset than the wins.

---

## Appendix material to consider

- **A. Full LB_HISTORY table** (already in `LB_HISTORY.md`; reproduce verbatim or summarize)
- **B. Per-version learning curves** (training loss + tr_auc for v19, v41, v44, v46)
- **C. Pseudo-label data flow diagram** (teacher → confidence threshold or raw probs → student training set)
- **D. Architecture diagram** (dual EffNet-B0 + MIL head)
- **E. Code release link** (your GitHub repo)

---

## Suggested figures (priority-ordered)

1. **LB progression line chart** (most important — instructor sees this first if they skim)
2. **Architecture diagram** (clean schematic, 1-column wide)
3. **Pseudo-label pipeline flowchart** (teacher predictions → confidence filter or raw → student training set merge)
4. **Hard vs soft pseudo histogram** (v44 vs v46: same teacher probabilities, but show how the 0.05/0.95 threshold discards 84% of cells)
5. **Learning curves grid** (3×3: rows = v19/v41/v46, cols = loss/tr_auc/epoch time)

## Suggested tables (priority-ordered)

1. **Version progression with LB + lesson** (centerpiece, from §6.2)
2. **Ablation: hard vs soft pseudo cell counts and LB** (single small table making the dataset-size argument)
3. **Cross-recipe ensemble attempts** (v22 + v45_probe, with members' LBs and result, showing the 0.02-gap rule)

---

## Writing-flow checklist

- [ ] Lead each section with the *insight*, then back it with *numbers* and *citations*
- [ ] Use the actual LB numbers from `LB_HISTORY.md` (not approximations)
- [ ] Cite: Lee 2013 (pseudo-labels), Hinton 2015 (distillation), Xie 2020 (noisy student), Izmailov 2018 (SWA), Li 2016 (AdaBN), Tan & Le 2019 (EfficientNet), Lian 2024 (the paper that inspired v37/v38)
- [ ] Mention what was tried and **why it didn't work** (negative results section is your strongest differentiator)
- [ ] Don't claim 0.85 is a target — re-read the instructor's May 14 announcement and align the framing to "what is done" rather than "score reached"
- [ ] Mention the GitHub repo with reproducible notebooks (you already pushed v39–v47)
