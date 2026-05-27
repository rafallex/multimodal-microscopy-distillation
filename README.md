# A3-ADL — Multimodal Cancer Classification Challenge 2026

Coursework for *Advanced Deep Learning for Image Processing* (1MD042), Uppsala University, Spring 2026. Binary cell-level cancer classification on paired bright-field (BF) + fluorescence (FL) microscopy at 128×128 grayscale resolution, with 12 training patients and a strict patient-disjoint test set.

**Current public-LB position: #3** (snapshot 2026-05-27). v47 ensemble at **0.8264**, v47 best single seed (v47_s2) at **0.8355**. Two teams overtook us between 2026-05-26 and 2026-05-27 (Group 1 at 0.8448, Group10 at 0.8445); our 0.8355 placement comes from v47_s2, not the v47 ensemble — empirical evidence for the within-recipe ensemble collapse documented in the paper §VII-G. Cumulative lift over the v19 supervised baseline is **+0.081** (ensemble) / **+0.090** (best seed) across 30 logged iterations.

## What's here

| Path | Description |
|---|---|
| [`overleaf-report/`](overleaf-report/) | **IEEE conference paper** (main.tex, refs.bib, four embedded PDF figures). Drop-in upload to Overleaf. `notes/` holds the local section drafts; `figure-sources/` holds the build scripts (each writes PNG for the deck + PDF for the paper in one invocation). |
| [`presentation/`](presentation/) | **In-class deck** (`A3_cancer_challenge_claude.pptx` is the lead version; the python-pptx build is kept as backup). PNG figures live in `figures/`, regenerated from `overleaf-report/figure-sources/`. |
| [`LB_HISTORY.md`](LB_HISTORY.md) | Every Kaggle submission in chronological order: recipe diff, public LB, one-line lesson. Includes the seven diagnosed negative-result analyses (v22, v27, v37/v38, v42, v43, v45_probe, v47 ensemble-vs-outlier). |
| [`REPORT_OUTLINE.md`](REPORT_OUTLINE.md) | Original drafting guide for the report — structured around the experimental narrative with hooks for each negative-result section. |
| `notebooks/` | One source notebook per iteration (`improvedvNN_source.ipynb`). Designed to run end-to-end on a Kaggle T4 ×2 in ~5–6 hours. |
| `results/` | Per-version submission CSVs (`v19`, `v41`, `v44`, `v46`, `v47` are committed for reproducibility, including per-seed extracts; intermediate training artifacts are gitignored). |

## Headline approach

The winning recipe is **EfficientNet-B0 dual-branch (BF + FL) with late concat fusion**, trained with patient-grouped MIL auxiliary loss, AdaBN + test-stain normalization, 3-seed × SWA ensembling, and 40-way TTA. The decisive lift came from a **semi-supervised pipeline**:

1. **v44 — hard pseudo-labels (Lee 2013).** Teacher v41 at threshold 0.05 / 0.95 keeps ~9,400 confident test cells. **+0.025 LB** over v41.
2. **v46 — soft-target distillation (Hinton 2015).** Teacher v44_seed1, all 59,040 test cells, raw probabilities as BCE targets. **+0.039 LB** — the largest single-experiment gain.
3. **v47 — iterative noisy student (Xie 2020), round 2.** Teacher swapped to v46 ensemble; otherwise identical to v46. **+0.003 LB on the ensemble**, +0.012 on the best single seed. The mean lift compressed ~13× from round 1 while the per-seed public-LB dispersion *expanded* ~2.8× (v46 range 0.0072 → v47 range 0.0205). The v47 ensemble at 0.8264 sits 0.0091 below the best seed (0.8355) — within-recipe averaging hedged against an outlier. Noise-floor analysis: the +0.003 ensemble lift is 0.31σ (indistinguishable from zero); the +0.012 best-seed lift is 1.33σ (marginal).
4. **v48 — Hinton temperature distillation (T=2).** Single-knob test on top of v47 to check whether the actual softening prescribed by Hinton 2015 buys anything beyond raw-probability soft pseudo. Queued for the 2026-05-29 Kaggle quota reset.

See [`LB_HISTORY.md`](LB_HISTORY.md) for the full progression and the seven negative-result diagnoses, and [`overleaf-report/main.tex`](overleaf-report/main.tex) for the IEEE paper write-up (§VII covers the negative results, §VIII-E discusses the noise floor and limitations).

## Reproducing a run

1. Open one of the source notebooks (e.g., `notebooks/improvedv47_source.ipynb`) on Kaggle.
2. Attach the required Kaggle inputs (listed in the notebook header markdown cell):
   - `rafaelproena/a3-adl` — competition data
   - A pseudo-label dataset for runs that use distillation (e.g., `ensamble-result-v46` for v47)
3. **Save Version → Save & Run All.** The notebook handles caching, training, SWA, AdaBN, TTA, and writes ensemble + per-seed submission CSVs to `/kaggle/working/`.

Compute budget: a single Kaggle T4 ×2 commit run, ~5–6 hours per pseudo-label version.

## Course context

Grades for this assignment are based on **methodology and presentation**, not on the leaderboard score itself (per instructor announcement). The experimental record in `LB_HISTORY.md`, the seven failure-mode diagnoses in the paper §VII, and the noise-floor discussion in §VIII-E are written with that grading framework in mind.

## License

MIT — see [`LICENSE`](LICENSE).
