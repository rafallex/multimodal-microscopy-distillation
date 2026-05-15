# How to read the notebooks

There are 14 notebooks in this repo, which is a lot. This document is a roadmap: it tells you which ones matter, what each version contributed, and which cells inside v16 are the "interesting" ones if you only have 10 minutes.

---

## File-naming conventions

| Suffix | Meaning | Typical size |
|---|---|---|
| `oral_cancer_baseline.ipynb` | The course-provided starter. Reference point only. | ~21 KB |
| `improvedvN_source.ipynb` | **Editable** notebook for version N. No embedded cell outputs. This is what you read to understand the method. | ~30–60 KB |
| `improvedvN_run.ipynb` | The same notebook **after a committed run on Kaggle**, with cell outputs and figures embedded. This is what you read for evidence that it actually executed. | ~180–240 KB |
| `improvedvN_colab_*.ipynb` | A **Colab variant** of version N. Paths point at Google Drive instead of `/kaggle/input`. Used as a fallback when Kaggle GPU quota is exhausted. | varies |

So `improvedv13_source.ipynb` and `improvedv13_run.ipynb` are the **same** version 13 method — one is editable, one is the run output.

The Colab variants are separate code paths because the data-access plumbing differs (Drive mount or `gdown` instead of `/kaggle/input`). Everything from "load dataframe" onward is identical between the Kaggle and Colab variants.

---

## Version timeline — what's worth reading

If you only have **10 minutes**, read just `improvedv16_source.ipynb`. It's the current version and contains the whole pipeline.

If you have **30 minutes**, read v15 then v16. v15 introduced the CoMIR SSL pretraining that v16 builds on, and the v15 → v16 diff is the substance of the report.

If you have **a full session**, read in this order:

| Version | Why read it | Key cell IDs to focus on |
|---|---|---|
| `oral_cancer_baseline` | See where we started — a single-modality ResNet-18 with no patient-grouped CV. | The whole thing, it's short. |
| `improvedv10` | First "improved" version. Introduces two-branch BF+FL architecture and patient-grouped CV. | `v10-model` |
| `improvedv11`–`v12` | Iterative additions: AdamW, OneCycleLR, paired geometric aug, patient-balanced sampler. | `v11-train`, `v12-aug` |
| `improvedv13` | Adds in-memory JPEG cache (cuts epoch time from 5 min to 2 min) and TTA at inference. | `v13-cache`, `v13-predict` |
| `improvedv14` | First mixup attempt. Turns out to fight the SSL features that v15 will introduce. | `v14-mixup` |
| `improvedv15` | **CoMIR-style contrastive SSL pretraining** + discriminative LR. The first version that targets cross-modal alignment directly. CV 0.866, LB 0.572 — the OOD gap that motivates v16. | `v15-ssl`, `v15-train-fn` (look for `MIXUP_ALPHA`) |
| `improvedv16` | **Current.** Nine changes from v15. See section below for the cell-by-cell tour. | All of it, but especially `v16-train-fn` |

---

## Reading `improvedv16_source.ipynb` cell by cell

v16 has 14 cells, each with a stable `id` so you can navigate by name:

| Cell ID | What it does | Why it matters |
|---|---|---|
| `v16-title` | Markdown overview of the nine changes from v15. | Read this first. It's the design summary. |
| `v16-imports` | `import torch, ...` plus a print of GPU info. | Skim. Standard setup. |
| `v16-config` | All hyperparameters as module-level constants (EPOCHS, BATCH_SIZE, LRs, SSL settings, SNAPSHOT_EPOCHS, EMA_DECAY, AUX_ALIGN_WEIGHT, FULL_DATA_SEEDS, ...). | **Read carefully.** Every design decision lives here as a one-line constant with a comment. |
| `v16-dataset` | `CachedCellDataset` and `ContrastivePairDataset` classes. | Read once to understand the dataset shape. JPEGs are decoded on-the-fly from a RAM byte cache. |
| `v16-splits-sampler-transforms` | LOPO splitter, `PatientBalancedSampler`, `RandomGamma`, `PairedGeoAug`, the three transform factories (train / eval / SSL). | The `ssl_modality_transform` is where the heavy stain aug lives — that's change #3. |
| `v16-model` | `_make_resnet18_branch`, `_make_proj_head`, `MultimodalClassifier` (with both `forward` and `forward_with_proj`), `ContrastiveModel`, `nt_xent_loss`. | The `forward_with_proj` method is the new piece for v16 — that's how change #9 (aux NT-Xent) is wired. |
| `v16-profile-and-cache` | Loads the CSVs, caches all train+test JPEG bytes into RAM dicts. | Skim. Single execution at the start of training. |
| `v16-ssl` | Stage 1: CoMIR contrastive SSL pretraining. ~25 min. Saves backbones + projection heads to `ssl_comir_backbone.pt`. | The heavy stain aug (#3) lives in `ssl_modality_transform` from the previous cell. |
| `v16-train-fn` | Defines `load_ssl_branches`, `ModelEMA`, `run_epoch`, `train_one_lopo_fold`, `train_full_data`. **This is the heart of v16.** | **Read carefully.** Changes #1 (no mixup), #6 (label smoothing), #7 (snapshots), #8 (EMA), #9 (aux NT-Xent) are all here. The `forward_with_proj` + frozen projections setup is in `make_discriminative_optimizer`. |
| `v16-train-lopo` | Stage 2: 12-fold LOPO supervised loop. ~4h 20min. Skips folds whose snapshot files already exist (resume-friendly). | The loop is trivial; the interesting stuff is what it calls (`train_one_lopo_fold`). |
| `v16-train-fulldata` | Stage 3: full-data model × 2 seeds. ~44 min. | Multi-seed full-data is change #9b. |
| `v16-curves` | Plots per-fold learning curves: train loss, train AUC, **held-out mean prediction over epochs** (color-coded by held-out label). | The third panel is the OOD diagnostic — red curves should trend toward 1, blue toward 0. Fold-level OOD failure is visible here. |
| `v16-oof` | Aggregates per-fold OOF predictions and reports cell-level + patient-level OOF AUC, plus per-snapshot AUC. | The per-snapshot AUC tells you whether ep3 or ep5 was more informative — guides v17's `SNAPSHOT_EPOCHS` if needed. |
| `v16-predict` | Stage 4: TTA inference across all ckpts, builds 4 submission CSVs (mean-of-logits, rank-avg, 0.5+0.5 mix, patient-aggregated). ~2h. | The hedge against patient-level LB scoring (#9c) is at the bottom. |

---

## Reading the Colab variant

`improvedv16_colab_source.ipynb` is **15 cells**, not 14. The extra cell at the top is `v16-colab-setup`:

- Mounts no Drive directly (because Drive auth doesn't work cleanly from VSCode-connected Colab)
- Installs `gdown` and downloads the competition zip from a public Drive share link
- Extracts to `/content/data/` (local SSD)
- Sets `COLAB_DATA_ROOT` and `COLAB_OUT_DIR` globals that downstream cells read

Everything else after the setup cell is **identical** to the Kaggle version (modulo `DATA_ROOT = COLAB_DATA_ROOT` and `OUT_DIR = COLAB_OUT_DIR` in the config cell).

If you want to verify they really are identical: run `diff` on the cells between `improvedv16_source.ipynb` and `improvedv16_colab_source.ipynb`. Only the title cell, setup cell (only present in Colab variant), and config cell should differ.

---

## What's NOT in the notebooks

- **The dataset itself** — gitignored. Fetch from Kaggle or the Drive share link.
- **Trained model weights** (the 39 v16 snapshot `.pt` files, ~90 MB each) — too large to commit. Will live in Kaggle/Colab output once v16 finishes.
- **The v15 model weights** — same reason. The `v15_baseline/` folder has history JSONs and OOF predictions only.

---

## If something is unclear

- The design reasoning lives in [`docs/design-decisions.md`](design-decisions.md).
- The end-to-end pipeline diagram lives in [`docs/methods-diagram.md`](methods-diagram.md).
- The v15 evidence that motivates v16 lives in [`v15_baseline/README.md`](../v15_baseline/README.md).
- The modular Python pipeline (from an earlier baseline) lives in [`code/README.md`](../code/README.md). It's a parallel implementation, not used by v16 itself.
