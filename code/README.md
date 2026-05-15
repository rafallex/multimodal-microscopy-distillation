# Multimodal Cancer Classification Challenge 2026 — baseline

## Files

| file              | what it does                                                         |
| ----------------- | -------------------------------------------------------------------- |
| `dataset.py`      | PyTorch `Dataset` that pairs BF + FL images; parses patient id.      |
| `splits.py`       | Patient-grouped K-fold + leave-one-patient-out CV splitters.         |
| `model.py`        | Two-branch ResNet-18 multimodal classifier (BF + FL).                |
| `transforms.py`   | Per-modality normalization + paired geometric augs.                  |
| `train.py`        | Training loop with AUC tracking + best-AUC checkpointing.            |
| `predict.py`      | Inference + averaged TTA + submission CSV writer.                    |
| `profile_data.py` | One-shot dataset profiler — run this once before training.           |

## Running on Kaggle (recommended)

1. Make a new notebook from the competition page. Data auto-mounts at
   `/kaggle/input/multimodal-cancer-classification-challenge-2026/`.
2. Settings: Accelerator = `GPU T4 x2`, Internet = On.
3. In the first cell, upload these `.py` files (or paste them as cells).
   Easiest: zip the `code/` folder, upload as a Kaggle Dataset, then
   `!cp -r /kaggle/input/<your-dataset>/code /kaggle/working/`.
4. Profile:
   ```bash
   !cd /kaggle/working/code && python profile_data.py \
       --data-root /kaggle/input/multimodal-cancer-classification-challenge-2026
   ```
5. Train one fold (≈ 8–15 min on T4):
   ```bash
   !cd /kaggle/working/code && python train.py \
       --data-root /kaggle/input/multimodal-cancer-classification-challenge-2026 \
       --out-dir   /kaggle/working/runs \
       --fold 0 --n-splits 5 --epochs 10 --batch-size 128 --num-workers 4
   ```
6. After training all 5 folds, predict on test:
   ```bash
   !cd /kaggle/working/code && python predict.py \
       --data-root /kaggle/input/multimodal-cancer-classification-challenge-2026 \
       --ckpts /kaggle/working/runs/fold0_best.pt \
               /kaggle/working/runs/fold1_best.pt \
               /kaggle/working/runs/fold2_best.pt \
               /kaggle/working/runs/fold3_best.pt \
               /kaggle/working/runs/fold4_best.pt \
       --out   /kaggle/working/submission.csv --tta
   ```
7. "Save Version" → "Save & Run All" → submit `submission.csv` from the
   output tab.

## Sanity checks before submitting

- `submission.csv` has exactly the same `Name` column as `sampleSubmission.csv`
  (order can differ; Kaggle matches by Name).
- Predictions are real numbers in [0, 1] — not 0/1 labels.
- Run `head submission.csv` to confirm the header is `Name,Diagnosis`.

## Improvements over the ResNet-18 baseline

| Change | Where | Expected effect |
|--------|-------|-----------------|
| EfficientNet-B0 backbone (`--backbone efficientnet_b0`) | `model.py` | Stronger features, better generalisation |
| Random 90° rotation steps in `PairedGeoAug` | `transforms.py` | Cells have no canonical orientation |
| Stronger ColorJitter (0.3) + RandomErasing | `transforms.py` | Reduces texture overfitting |
| Mixup (alpha=0.2) | `train.py` | Smoother decision boundary |
| Label smoothing (eps=0.05) | `train.py` | Reduces overconfident logits |
| Gradient clipping (max_norm=1.0) | `train.py` | Stable EfficientNet training |
| OneCycleLR with 10% warmup | `train.py` | Faster convergence |
| Early stopping (patience=6) | `train.py` | Saves compute, avoids overfitting |
| DataParallel (auto) | `train.py` | Uses both T4 GPUs |
| 8-way D4 TTA (4 rotations × 2 reflections) | `predict.py` | Better test-time prediction |

## Phased plan

1. **Improved baseline (this code)**: EfficientNet-B0 × 2 branches, full D4 augmentation,
   mixup, label smoothing. Target ≥ 0.87 AUC.
2. **Ablations**: BF-only, FL-only, both — quantify the multimodal gain.
3. **SSL pretraining**: SimCLR or MAE on the *combined* train+test images
   (we have labels only for train, but the *images* of test are usable).
4. **MIL / patient-level aggregation**: aggregate per-patient predictions and
   use that as an extra feature, or average per-patient at test time.
5. **Ensembling**: average multiple architectures and seeds.

## Key gotchas in this dataset

- **Weak labels.** Every cell of a cancer patient is labeled 1, even though
  most individual cells are visually indistinguishable from benign ones. The
  AUC ceiling is not 1.0; the model is learning population-level signal.
- **Only 19 patients.** Fold-to-fold AUC variance will be large. Always
  report mean ± std across folds, never a single fold.
- **Don't split cells randomly.** Always group by patient_id. The filenames
  encode it for a reason.
- **Class imbalance.** Use `pos_weight` in BCE, or oversample, or focal loss.
