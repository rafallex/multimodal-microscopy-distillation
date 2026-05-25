# A3-ADL — Multimodal Cancer Classification Challenge 2026

Coursework for *Advanced Deep Learning for Image Processing* (1MD042), Uppsala University, Spring 2026. Binary cell-level cancer classification on paired bright-field (BF) + fluorescence (FL) microscopy at 128×128 grayscale resolution, with 12 training patients and a strict patient-disjoint test set.

**Current public-LB position: #1 with 0.8236** (29 logged iterations, +0.078 LB lift over the v19 baseline).

## What's here

| Path | Description |
|---|---|
| [`LB_HISTORY.md`](LB_HISTORY.md) | Every Kaggle submission in chronological order: recipe diff, public LB, one-line lesson. Includes negative-result analyses for v22, v27, v37/v38, v42, v43, and v45_probe. |
| [`REPORT_OUTLINE.md`](REPORT_OUTLINE.md) | Drafting guide for the final A3 report — structured around the experimental narrative with explicit hooks for each negative-result section. |
| `notebooks/` | One source notebook per iteration (`improvedvNN_source.ipynb`). Designed to run end-to-end on a Kaggle T4 ×2 in ~5–6 hours. |
| `presentation/` | Builder scripts for the in-class presentation deck. |
| `results/` *(gitignored)* | Local-only training artifacts: per-seed history JSON, SWA checkpoints, submission CSVs. |

## Headline approach

The winning recipe is **EfficientNet-B0 dual-branch (BF + FL) with late concat fusion**, trained with patient-grouped MIL auxiliary loss, AdaBN + test-stain normalization, 3-seed × SWA ensembling, and 40-way TTA. The decisive lift came from a **semi-supervised pipeline**:

1. **v44** — pseudo-labels (Lee 2013) from a v41 teacher at threshold 0.05 / 0.95: +0.025 LB.
2. **v46** — soft-target distillation (Hinton 2015) using all 59,040 test cells with raw teacher probabilities: **+0.039 LB** (biggest single-experiment gain).
3. **v47** — iterative noisy-student round 2 (Xie 2020) using v46 as the new teacher: in progress.

See `LB_HISTORY.md` for the full progression and `REPORT_OUTLINE.md` for the methodological narrative.

## Reproducing a run

1. Open one of the source notebooks (e.g., `notebooks/improvedv47_source.ipynb`) on Kaggle.
2. Attach the required Kaggle inputs (listed in the notebook header markdown cell):
   - `rafaelproena/a3-adl` — competition data
   - A pseudo-label dataset for runs that use distillation (e.g., `ensamble-result-v46` for v47)
3. **Save Version → Save & Run All**. The notebook handles caching, training, SWA, AdaBN, TTA, and writes ensemble + per-seed submission CSVs to `/kaggle/working/`.

Compute: a single Kaggle T4 ×2 commit run, ~5–6 hours per version.

## Course context

Grades for this assignment are based on **methodology and presentation**, not on the leaderboard score itself (per instructor announcement). The experimental record in `LB_HISTORY.md` and the analytical scaffolding in `REPORT_OUTLINE.md` are written with that grading framework in mind.

## License

MIT — see [`LICENSE`](LICENSE).
