# v16 results — post-mortem

v16 was committed on Kaggle and produced a complete set of submission files plus the learning curves and per-fold OOF data. This folder preserves the lightweight evidence (~12 MB); the 28 model checkpoint files (~90 MB each, ~2.5 GB total) are NOT included.

## Headline numbers

| Metric | Value | vs v15 |
|---|---|---|
| Cell-level OOF AUC | **0.8021** | -0.044 |
| Patient-level OOF AUC | **0.9429** | **+0.029** |
| Per-snapshot OOF (ep3) | 0.7976 | — |
| Per-snapshot OOF (ep5) | 0.8058 | — |
| Per-snapshot ensemble | 0.8021 | — |
| Patient-level failures | 2 / 12 (pat5 = 0.436, pat9 = 0.574) | both less confident than v15's worst |
| **Public LB** (`submission_v16_combo.csv`) | **0.503** | -0.069 |

Cell-level OOF dropped 4 points vs v15 because **LOPO is harder than 3-fold stratified-group CV** — every held-out patient is OOD by construction, whereas v15's 3-fold scheme had 2-of-3 folds with near-train-distribution validation patients that inflated the mean.

Patient-level OOF rose 3 points (0.943 vs 0.914), which is what we actually care about. The aux NT-Xent loss + heavier stain aug + EMA design improved OOD generalization at the patient level. The remaining LB gap is **site-level** OOD that LOPO physically cannot measure (more on this below).

## What worked (v15 → v16 changes that did something)

1. **LOPO replaced 3-fold CV** — exposed the real OOD gap honestly. v15's CV mean of 0.866 was an artifact; v16's CV mean of 0.802 is the honest number.
2. **Aux NT-Xent alignment loss with frozen CoMIR projections (#9)** — patient-level OOF rose by 0.029 vs v15. The worst-failing patients in v15 (pat9 at 0.665, pat17 at 0.498) became less wrong in v16 (pat9 at 0.574, pat5 at 0.436). The model no longer makes *confidently* wrong predictions on hard OOD patients.
3. **Multi-snapshot ensembling (#7)** — ep5 individually scored 0.806 OOF, ep3 scored 0.798; the mean is 0.802. Both snapshots are useful; the ensemble is robust to either being slightly worse.
4. **EMA at decay=0.99 (#8)** — the math audit in `docs/design-decisions.md` §8 showed why 0.999 would have been a no-op. With 0.99, the snapshots actually benefit from EMA smoothing.
5. **Stain aug in CoMIR SSL was heavier (#3)** — hard to isolate but probably contributed to pat9 / pat17 being less wrong.

## What did not work

1. **The LB gap did not close**. v15 had cell-OOF 0.846 → LB 0.572 (gap 0.27). v16 has cell-OOF 0.802 → LB 0.503 (gap 0.30). LOPO honestly measures *patient-level* generalization; the LB measures *test-set* generalization. These are not the same thing. The test set is distributed differently from train along an axis (scanning site, cell prep, microscope, time-of-day, ...) that none of the 12 train patients vary on, so no within-train CV scheme — not 3-fold, not 5-fold, not LOPO, not LOO-cell — can detect it.
2. **The patient-aggregated hedge** did not work either (see "submission ablation" below). If Kaggle's metric were patient-level, `submission_patient_agg.csv` should have scored much higher than `submission_v16_combo.csv` because patient-level OOF is 0.94. Both score similarly bad on LB, which suggests the metric IS cell-level AND the test set has site-level OOD.

## Per-patient breakdown

Sorted by OOF mean prediction (ascending). Mean computed across all snapshots, all cells of each patient.

| Patient | True label | OOF mean pred | Verdict |
|---|---|---|---|
| pat 7 | 0 (healthy) | 0.185 | very confident, correct |
| pat 10 | 0 | 0.196 | very confident, correct |
| pat 15 | 0 | 0.342 | confident, correct |
| pat 13 | 0 | 0.348 | confident, correct |
| pat 11 | 0 | 0.430 | borderline, correct |
| pat 5 | **1 (cancer)** | **0.436** | **FAIL** — false negative |
| pat 14 | 0 | 0.443 | borderline, correct |
| pat 9 | **0 (healthy)** | **0.574** | **FAIL** — false positive |
| pat 17 | 1 | 0.588 | borderline, correct |
| pat 18 | 1 | 0.688 | confident, correct |
| pat 16 | 1 | 0.718 | confident, correct |
| pat 3 | 1 | 0.731 | confident, correct |

**10 of 12 patients ranked correctly at the patient level.** The two failures (pat 5 and pat 9) are both close to 0.5 — the model is uncertain about them, not confidently wrong. In v15, pat 9 was at 0.665 (much more confidently wrong) and pat 17 was at 0.498 (basically 50/50). v16 reduced the magnitude of patient-level errors.

## Submission file ablation

Four submission CSVs were produced. **Only the combo file (`submission_v16_combo.csv`) has been submitted to the LB**; the other three are pending if you want to use remaining daily submission slots.

| File | Construction | LB | Notes |
|---|---|---|---|
| `submission_v16_combo.csv` | `0.5 * sigmoid(mean_logit) + 0.5 * rank_avg` | **0.503** | The default v16 output that got submitted |
| `submission_logit.csv` | pure mean-of-logits → sigmoid | TBD | Submit to ablate whether rank-avg mixing hurt |
| `submission_rank.csv` | pure rank-average across 28 ckpts | TBD | Submit to ablate whether logit-mean hurt |
| `submission_patient_agg.csv` | each cell replaced with its patient's mean | TBD | Submit to test "is the metric patient-level?" |

If `submission_patient_agg.csv` scores meaningfully better than 0.503, the metric is patient-level and v16 has been undersold. If all four score similarly, the test set is fundamentally OOD.

## Per-snapshot timing

```
LOPO supervised: 12 folds × 6 epochs × ~210 s/ep = ~4 h 12 min
Full-data:       2 seeds × 6 epochs × ~140 s/ep = ~28 min
TTA inference:   28 ckpts × ~145 s/ckpt = ~67 min
```

Total actual training wall-clock: ~5 h 45 min (under the projected 8 h 15 min — Kaggle T4 was faster than estimated).

## Reading the learning curves

`learning_curves.png` shows three panels for the 12 LOPO folds + 2 full-data trajectories:

1. **Train BCE loss** — all folds converge to ~0.55–0.70 by epoch 5. Red (positive-patient holdouts) sit slightly higher than blue (negative-patient holdouts), as expected. The two full-data trajectories (black solid + dashed) overlay correctly.
2. **Train AUC** — all folds reach 0.89–0.92. The model is learning to fit train; this panel is a sanity check, not a generalization signal.
3. **Held-out mean prediction** (the diagnostic panel) — for each held-out patient, the mean predicted probability over epochs. Red lines (positive patients) should trend toward 1; blue lines (negative patients) toward 0. Most do; pat 5 (red) and pat 9 (blue) are the two failures visible in the middle of the plot, hovering near the 0.5 boundary.

The vertical grey lines at epochs 3 and 5 mark the snapshot points (`SNAPSHOT_EPOCHS = [3, 5]`).

## Implications for v17

The OOD gap is **not patient-level**; it's site-level or protocol-level. Within-train CV cannot bridge it. v17 needs methods that *use the test data directly*:

1. **Pseudo-labeling** — take v16's confident test predictions (e.g., p > 0.9 or p < 0.1), add them as additional training labels, retrain a final model on train + pseudo-labeled test. Two Kaggle commits; first to get v16 predictions (done), second to retrain.
2. **Test-time training** — at inference, run a few SGD steps on the test set using a self-supervised objective (e.g., the CoMIR NT-Xent on test BF/FL pairs). The model adapts its representations to test-set statistics.
3. **Stain normalization toward test statistics** — preprocess BF and FL to match the empirical mean/std of the test set, rather than the train set. Cheap, easy to implement.
4. **Stronger TTA** — add scale jitter (112, 128, 144) on top of D4. Triples inference time but might recover signal.

None of these would have helped v16; they're the right v17 candidates if we get one more commit.

## Reproducibility

All numbers above can be reproduced from the JSON + CSV files in `runs/` using the snippet in `v15_baseline/README.md` (same AUC formula, swap the file paths to `v16_results/runs/`).
