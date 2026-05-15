# Design decisions for v16

This document collects the formal design decisions made during the v15 → v16 transition. It captures the *reasoning*, not just the code change. Sections 3–5 follow the **ADR (Architecture Decision Record)** format — each ADR states the context, the options considered, the decision, and the consequences.

Audience: my future self (and the assignment report grader).

---

## 1. The big picture

v15 was the first version with CoMIR-style contrastive SSL. It produced the highest CV AUC of any version (0.866) and the lowest LB (0.572) — a gap of 0.29 AUC. Diagnosing that gap and closing it is the entire purpose of v16.

The root causes of v15's collapse, derived from `v15_baseline/runs/` and the v15 code review:

1. **3-fold CV hid the OOD failure.** Two of three folds had val patients similar to train patients (val AUC > 0.88). One fold had OOD-like val patients (val AUC 0.78–0.82, peaked at epoch 0). The CV mean (0.866) buried the failure.
2. **Mixup was fighting CoMIR.** The SSL pretext task trained the backbones to align paired (BF, FL) cells in feature space. Mixup then trained the model on synthetic *mixtures* of two cells, which have no real cross-modal correspondence — pulling the backbone away from its learned invariance.
3. **The supervised stage drifts away from the CoMIR alignment** because BCE only rewards classification, not invariance. With no anchor, the backbone slowly forgets what SSL taught it.
4. **The best-AUC-epoch checkpointing scheme** picked epoch 0–1 for hard folds (essentially the SSL init plus one gradient step) and epoch 3–5 for easy folds. The ensemble averaged these inconsistent training stages.

v16 addresses each cause directly. The nine changes in `README.md` map to these causes as follows:

| Cause | v16 fix |
|---|---|
| (1) 3-fold CV hides OOD failure | LOPO 12-fold CV (#2) |
| (2) Mixup fights CoMIR | No mixup during fine-tune (#1) |
| (3) Supervised drift from CoMIR | Aux NT-Xent on frozen projections (#9) |
| (4) Inconsistent checkpoint stages | Multi-snapshot ensemble at {3, 5} (#7) + EMA smoothing (#8) |

The other changes (#3 heavier stain aug, #4 logit-space TTA, #5 ensemble structure, #6 label smoothing) are independent improvements that hedge against secondary failure modes.

---

## 2. How to read the ADRs

Each ADR follows this structure:

- **Context** — what was the situation that forced the decision
- **Options considered** — the alternatives I weighed (not just the winner)
- **Decision** — what I picked and why
- **Consequences** — what becomes easier, what becomes harder, what to revisit

ADRs are numbered in the order they were made. ADR-001 is the earliest (after the v15 code review surfaced the OOD gap). ADR-003 is the latest (the night before the planned Kaggle commit).

---

## 3. ADR-001 — Which L4 lecture tips to add to v16

**Status:** Accepted
**Date:** 2026-05-14
**Decider:** Rafael

### Context

The v15 code review surfaced the OOD failure and produced a long list of potential fixes. To prioritize, I mined the course lectures (especially L4 on the learning process) for techniques that might apply. The lecture-mining produced eight candidate additions:

1. Best-snapshot tracking per fold
2. Label smoothing on BCE targets
3. Squeeze-and-Excitation blocks in ResNet-18
4. Focal Loss as alternative to BCE+pos_weight
5. Cosine annealing with warm restarts (SGDR)
6. Reflection padding instead of zero padding
7. t-SNE/UMAP of CoMIR features (diagnostic only)
8. Curve smoothing with hysteresis on val tracking

I had to decide which to actually include in v16, balancing expected gain against implementation risk and the risk of "tinkeritis" the day before a high-stakes Kaggle commit.

### Options considered

**Option A — Do nothing (ship v16 as designed)**
- Low complexity, no compute cost, zero risk
- But leaves known free improvements on the table

**Option B — Label smoothing only**
- One-line change to BCE targets (ε = 0.05)
- Cited by L4 p.48 ("if adding blur to the input is useful, why not to the output as well?")
- Zero compute cost, very low risk
- Helps the rank-averaged ensemble because smoothed targets reduce over-confident logits

**Option C — Label smoothing + best-snapshot tracking with directional score**
- Adds snapshot-selection logic during LOPO training
- The directional score `s = held_mean if held_class==1 else 1-held_mean` measures how confidently the model puts the held-out patient toward the correct label
- Untested under LOPO; could bias selection toward folds that overfit to held-out stain

**Option D — Label smoothing + cosine annealing with warm restarts (SGDR)**
- Replaces OneCycleLR with 2 × 3-epoch SGDR cycles
- L4 explicitly endorses this scheduler (the lecturer's stated favorite)
- Saves end-of-cycle weights as a free snapshot ensemble
- Risk: short cycle length (3 epochs) is off-piste; standard SGDR uses ≥10-epoch cycles

### Decision

**Adopted Option B: label smoothing only.** Defer #C and #D to a potential v17.

### Reasoning

- **Why B over A:** Label smoothing is a free, well-understood win. Calibration matters specifically because v16's final submission is `0.5 × sigmoid(mean_logit) + 0.5 × rank_avg`; reducing logit saturation helps the first term align better with the second. No downside.
- **Why B over C:** The directional score is unsound on a single-class held-out patient. Selecting on `held_mean → 1` for positive patients literally rewards confident overfitting to that patient's stain — the opposite of what LOPO is supposed to enforce. The lecture's "best on val" advice assumes val AUC is well-defined; under LOPO it isn't, and substituting a proxy that misrepresents the OOD problem is worse than just using the final epoch (which OneCycleLR has already cooled to a stable minimum).
- **Why B over D:** The expected upside is real but the risk is unbounded. Short 3-epoch SGDR cycles are off-piste; the lecturer's example assumes longer schedules. No time for an ablation. If v16 lands well, this is the right candidate for v17.

### Consequences

- **Easier:** v16 ships with one extra defensible tip, traceable to L4 p.48. Report writes itself.
- **Harder:** If v16's LB is still bad, we'll need a v17 with bigger swings (cosine restarts, multi-seed LOPO, or a different backbone).
- **Revisit when:** v16 LB is in. If >0.75 we stop. If 0.60–0.75 we try D for v17. If <0.60, the problem isn't optimization-side at all and we need to revisit the SSL pretext task itself.

---

## 4. ADR-002 — Pre-commit go/no-go for v16

**Status:** Accepted
**Date:** 2026-05-15
**Decider:** Rafael

### Context

v16 was implemented, code-reviewed twice, and tuned to fit a ~8 h 30 min budget. About to click "Save & Run All" on Kaggle. One commit attempt costs ~9 h of wall-clock and one daily competition submission slot.

I needed a single sanity pass to catch the kind of bug that wastes the whole run — not to redesign anything.

### Options considered

**Option A — Commit now**
- Fastest path to the LB number that decides whether v17 is needed
- Code has been audited line-by-line
- 30 min headroom of 9 h is thin but defensible

**Option B — Run a 1 h "smoke test" first** with `EPOCHS = 1, SNAPSHOT_EPOCHS = [0]`
- Catches setup errors before the real run
- Doubles wall-clock cost (1 h smoke + 8.5 h real)
- The code's been reviewed twice — marginal probability of a setup-only bug is low

### Decision

**Commit as-is, but verify the 5 pre-commit settings below first and watch the 4 mid-run checkpoints.**

### Pre-commit checklist (final, before clicking Save Version)

- [ ] Settings → Accelerator = **GPU T4 ×2**
- [ ] Settings → Internet = **On**
- [ ] Input panel → `multimodal-cancer-classification-challenge-2026` mounted at `/kaggle/input/competitions/...`
- [ ] Notebook tab → `improvedv16_source.ipynb` (title cell says "Nine changes from v15")
- [ ] First cell output → `DATA_ROOT = /kaggle/input/...` with the assert passing

### Mid-run checkpoints (rough phone timers)

- **~50 min in** — cache step finishes with "Approx RAM used by JPEG cache: ~500 MB" (if >14 GB, abort)
- **~1 h 20 min in** — SSL stage ep 4 loss is < 1.0 (NT-Xent typically lands 0.4–0.8); abort and lower stain aug if NaN or stuck > 2.0
- **~3 h 30 min in** — at least 6 LOPO folds done
- **~6 h in** — all LOPO + full-data done, TTA inference has started

### Consequences

- **Easier:** Anchored numerical expectations for what "healthy" intermediate output looks like. If something deviates, abort early and save the next commit attempt.
- **Harder:** If the abort happens at hour 7, the partial training ckpts persist in `/kaggle/working/runs/` but the inference step won't have run. Would need a follow-up "inference-only" commit.

---

## 5. ADR-003 — Maximize v16 LB during the 24 h Kaggle quota wait

**Status:** Accepted
**Date:** 2026-05-15
**Decider:** Rafael

### Context

The Kaggle GPU quota was exhausted before v16 could be committed. The quota would reset in ~24 hours. The user wanted to use that time to make v16 stronger before the (now-rescheduled) commit. The question: what additions can be made *during the wait* that would meaningfully improve the LB, without destabilizing the existing design?

Forces and constraints:
- 9 h hard commit limit; current budget is 8 h 30 min
- One commit = one shot per ~30 h quota window (no fast iteration)
- Risky changes that fail at commit time waste the whole day
- The CoMIR SSL backbone is the most expensive piece; preserve it, don't invalidate it
- The user values upside over safety this round

### Options considered

**Option A — Minimal safe additions (EMA only)**
- Add an EMA wrapper around the supervised-training model
- Save EMA weights at snapshot points instead of raw weights
- Established trick, almost always nets a small win
- Zero compute cost added

**Option B — A + multi-seed full-data + patient-aggregated submission**
- Run the full-data model with 2 random seeds (more ensemble diversity)
- Write a parallel `submission_patient_agg.csv` that replaces each cell's prediction with its test-patient mean (free hedge against patient-level LB scoring)
- +35 min compute (fits if we drop snapshot_epochs from [3,4,5] to [3,5])

**Option C — B + aux SSL-alignment loss during supervised** *(recommended)*
- Retain the CoMIR projection heads from SSL, freeze them, and add `0.05 × NT-Xent(z_bf, z_fl)` to BCE
- Backprops through the (unfrozen) backbones, anchoring them to the CoMIR objective
- Directly opposes the BCE-driven drift away from cross-modal invariance — the exact mode that caused v15's LB collapse
- New code path; medium risk

### Decision

**Adopted Option C** with `SNAPSHOT_EPOCHS = [3, 5]` to free budget.

### Reasoning

The aux NT-Xent loss is the only one of these ideas that *changes the gradient direction* of the supervised stage. EMA and multi-seed only re-sample what the gradient already produces; they help but they can't fix a misdirected optimizer. The user wants LB dominance, which on this dataset means winning the OOD gap. Aux NT-Xent is the only tool here that directly does that.

Risk mitigation: keep the aux weight small (0.05), add a sanity-check that prints `bce_loss` and `aux_loss` separately for the first epoch (catches sign errors or scaling issues), and if anything looks off the aux weight can be set to 0 by flipping one constant — degrading C to B with zero rework.

### Consequences

- **Easier:** v16 ships with three asymmetric-upside additions and one safety hedge (patient-agg). Each is motivated by a separate failure mode of v15.
- **Harder:** Three places where a bug could land instead of one. Aux weight is untuned — if it dominates BCE, we get a CoMIR-like representation with bad classification.
- **Revisit when:** v16 LB lands. If >0.80 we stop and write the report. If 0.65–0.80 the gain came from somewhere and we ablate for v17. If <0.65, the aux loss probably destabilized something — set its weight to 0 and try again.

---

## 6. Supporting audit — Lecture tip mining

I scanned the course lectures (L1a, L1b, L3, L4, L5, L6, L8, L9) for techniques that could improve v16. The full audit:

### Tips applied in v16

| Tip | Lecture | Page | v16 location |
|---|---|---|---|
| Discriminative LR (1:10 backbone:head ratio) | L4 | p.96 | `BACKBONE_LR = 3e-5`, `HEAD_LR = 3e-4` |
| Label smoothing on BCE targets | L4 | p.48 | `LABEL_SMOOTHING = 0.05` |
| AdamW optimizer | L4 | p.45 | `torch.optim.AdamW` |
| Data augmentation including paired geometric | L4 | p.82 | `PairedGeoAug`, `ssl_modality_transform` |
| Transfer learning from large-data SSL | L4 | p.93–99 | CoMIR pretrain → supervised |
| Dropout | L4 | p.61 | `DROPOUT = 0.3` in head |
| BatchNorm | L4 | p.84 | Stock ResNet + head BN1d |
| No data leakage in CV | L4 | p.8 | Patient-balanced sampler + LOPO |

### Tips evaluated and skipped (with reasoning)

| Tip | Why skipped |
|---|---|
| SE-blocks (L4 p.26) | Changes ResNet-18 architecture, invalidates the SSL backbone weights. Would require re-pretraining. Cost > benefit. |
| Focal Loss (L3) | Our positive class is ~50% (12 patients, balanced ±1); class imbalance is mild. `pos_weight` already handles it. |
| Reflection padding (L3 p.33) | Would require monkey-patching every conv in ResNet. Tiny expected effect. |
| t-SNE of CoMIR features (L5 p.14–15) | Diagnostic only, doesn't move the LB. |
| Cosine restarts (L4 p.35) | See ADR-001 — deferred to v17 if v16 doesn't land well. |
| ViT (L9) | Explicitly needs JFT-300M-scale data. Our dataset is far too small; CNN inductive bias is correct here. |
| Model-based DL (L8) | Wrong modality (inverse problems / reconstruction). We're doing classification. |
| Best-snapshot tracking with proxy score (L4 p.41–42) | The directional score is unsound on a single-class held-out patient under LOPO — see ADR-001 Option C reasoning. |

---

## 7. Supporting audit — AMP numerical safety

I audited v16 for potential overflow / underflow under automatic mixed precision (AMP). The hypothesis worth testing: NT-Xent at τ = 0.1 with FP16 features could produce `exp(10)` ≈ 22,000 inside the loss, which is at the edge of FP16's range (~65,504).

### Why we're actually safe

PyTorch's autocast has an explicit FP32 cast policy for ops where FP16 is known-dangerous. The relevant ops on the FP32 list:

- `torch.nn.functional.cross_entropy`
- `torch.nn.functional.log_softmax`
- `torch.nn.functional.softmax`
- `torch.nn.functional.binary_cross_entropy_with_logits`

So when `F.cross_entropy(sim, targets)` is called inside an `autocast("cuda")` block:

1. The `sim` tensor (FP16, values in [-10, 10]) gets **cast up to FP32** before the op runs
2. `log_softmax` computes `sim_fp32 - max(sim_fp32)` in FP32 → range [-20, 0]
3. `exp(...)` in FP32 → [2e-9, 1.0] — no overflow, no meaningful underflow
4. Sum, log, gather — all FP32
5. Final loss is FP32, gets scaled by the GradScaler for backward

Same protection applies to `BCEWithLogitsLoss` for the supervised classification loss.

### Other AMP-adjacent things checked

| Concern | Status |
|---|---|
| Sim values overflowing FP16 BEFORE cross_entropy is called (during `z @ z.t() / tau`) | Safe — `sim` is FP16 but bounded [-10, 10], inside FP16's [-65504, 65504] range |
| `masked_fill_(eye, float("-inf"))` in FP16 | Safe — FP16 has a native `-inf` representation |
| BatchNorm in head with FP16 | Safe in train (drop_last=True → batch=128) and val (model.train(False) → running stats) |
| Aux loss gradient flow through frozen projections | Correct — `requires_grad=False` stops the param update but not the chain rule. Gradient reaches the backbone. |
| EMA shadow of frozen projection weights | Correct — shadow converges to the unchanging SSL value (`decay × shadow + (1-decay) × ssl_value → ssl_value`) |
| State-dict swap for EMA val pass preserving requires_grad flags | Correct — `load_state_dict` only copies `.data`, leaves `requires_grad` alone |
| Last batch of val_loader could be size 1 (no drop_last) | Safe — val runs with model.eval(), BN uses running stats |
| Single-class val patient under LOPO causes `roc_auc_score` to error | Handled — explicit `len(np.unique) > 1` check |

### Verdict

No bugs found. v16 is in shippable shape.

---

## 8. Supporting audit — EMA decay math

The first version of v16's EMA used `EMA_DECAY = 0.999`, the conventional value. After the AMP audit, I checked whether 0.999 was actually correct for our training length.

### The math

EMA shadow update: `shadow = decay × shadow + (1 - decay) × params`

After N steps from `shadow = params_initial`, with `params` changing each step, the fraction of "weight on the initial value" in the shadow is `decay^N`.

For v16's training: 12-patient train set, drop ~1 patient (LOPO), ~9100 cells × (11/12) per fold. Batch 128, drop_last → 71 batches/epoch × 6 epochs = **426 steps per fold**.

```
decay = 0.999 → 0.999^426 = 0.653   ← 65% of EMA shadow is still random init!
decay = 0.995 → 0.995^426 = 0.118   ← 12% — borderline
decay = 0.99  → 0.99^426  = 0.014   ← 1.4% — useful
decay = 0.95  → 0.95^426  ≈ 10^-10  ← essentially tracks current params
```

At snapshot time (ep 3 and ep 5), with `decay = 0.999`:
- ep 3 shadow ≈ 0.999^(71×3) = 0.999^213 ≈ **0.808** initial weight
- ep 5 shadow ≈ 0.999^(71×5) = 0.999^355 ≈ **0.701** initial weight

In other words: at `decay = 0.999`, the saved EMA snapshots would be 70–80% random init weights. EMA would be worse than useless — actively polluting the snapshots with pre-trained noise.

### The fix

Picked `EMA_DECAY = 0.99` based on what we want EMA to do:

| Value | Effective window (`1/(1-decay)`) | What it does |
|---|---|---|
| 0.99 | ~100 steps (~1.4 epochs) | Smooths OneCycleLR's late-stage oscillations |
| 0.995 | ~200 steps (~3 epochs) | Wider average, more lag |
| 0.999 | ~1000 steps (longer than training) | Effectively no-op |

### Lesson

EMA tuning is rarely discussed because the conventional 0.999 *works* on long training (ImageNet, BERT). For short training, it has to be scaled down. The rule of thumb: `1/(1-decay)` should be substantially smaller than your total training step count.

---

## 9. Open questions for v17

If v16 lands but the LB number isn't strong enough, these are the directions to consider next, roughly ranked by expected ROI:

1. **Pseudo-labeling.** Use v16's confident test predictions (e.g., p > 0.95 or p < 0.05) as additional training labels, retrain a final model on train + pseudo-labeled test. Requires 2 Kaggle commits but directly exploits the test distribution.
2. **Cosine annealing with warm restarts (SGDR).** Replace OneCycleLR with 2 × 3-epoch cycles, save end-of-cycle weights as additional snapshots. ADR-001 Option D.
3. **Multi-seed LOPO.** Run each LOPO fold with 3 random seeds instead of 1. Triples the ensemble size. Pricey on compute (~12 h instead of 8 h 30 min) — would need to drop something else to fit in 9 h.
4. **Backbone upgrade.** EfficientNet-B0 or ConvNeXt-Tiny would invalidate the SSL backbone but might be worth it if the limit really is the visual representation power. Big risk, big potential.
5. **Stronger SSL.** Increase CoMIR epochs from 5 to 10, increase batch size to 384 if memory allows. Bigger SSL → better features. ~25 min more compute.
6. **Two-stage TTA.** Add scale jitter (112, 128, 144) on top of D4. 3× TTA cost; would have to drop snapshots to fit in 9 h.

The cheapest of these to try first is #1 (pseudo-labeling). It's also the most likely to actually move the LB, because it's the only one that uses the *real test distribution* directly.

---

*Last updated: 2026-05-15. Will be appended to as v16 / v17 results land.*
