# overleaf-report/ — the Overleaf upload bundle

Everything paper-related lives in this single folder. To send the paper to
Overleaf, upload only the **6 files at the root of this folder** (the
`notes/` and `figure-sources/` subfolders below are paper-development
material and stay local).

## What goes to Overleaf

Upload these 6 files into a fresh Overleaf project (drag-and-drop):

| File | Role |
|---|---|
| `main.tex` | The paper (IEEE conference template, ~8 pages compiled). **This is the canonical source — edit it here.** |
| `refs.bib` | Bibliography (BibTeX format, 13 cited entries). |
| `arch_diagram.pdf` | Figure 1 (dual EfficientNet-B0 + MIL aux). |
| `pseudo_pipeline.pdf` | Figure 2 (v44 hard → v46 soft → v47 iterated). |
| `lb_progression.pdf` | Figure 3 (public-LB progression chart). |
| `teacher_prob_histogram.pdf` | Figure 4 (v46 teacher probability distribution). |

Then: Menu → Settings → **Compiler = pdfLaTeX**, **Main document = `main.tex`**
→ click **Recompile**. BibTeX runs automatically.

## What stays local (don't upload)

| Path | Role |
|---|---|
| `notes/06_results.md` | Long-form results draft (paper-development notes). |
| `notes/07_negative_results.md` | Long-form negative-results draft (paper-development notes). |
| `figure-sources/build_*.py` | Python scripts that regenerate the 4 figure PDFs (and the matching PNGs for the PPT in `../presentation/figures/`). |
| `README.md` | This file. |

## Local build (no Overleaf)

Requires a LaTeX distribution: MiKTeX (Windows) or TeX Live (macOS/Linux).
The relevant packages are `texlive-publishers` (for `IEEEtran.cls`) and
`texlive-fonts-recommended` (recommended T1-encoded fonts; the `fontenc`
package itself is part of the LaTeX base). MiKTeX installs missing packages
on-demand by default.

```
cd overleaf-report
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

Three pdflatex passes are required: the first builds `main.aux`, `bibtex`
reads it to write `main.bbl`, the second pass embeds the bibliography, and
the third pass resolves any remaining cross-references and page numbers.
Output: `main.pdf` in this folder.

## Figure paths

`main.tex` sets `\graphicspath{{./}}`. The four figure PDFs sit at the root
of this folder, next to `main.tex`. This works for both Overleaf (after the
flat upload) and a local build from inside `overleaf-report/`.

## Regenerating figures

The figure PDFs in this folder are produced by the scripts in
`figure-sources/`. Each script writes **two** outputs:

- `overleaf-report/<name>.pdf` — consumed by the LaTeX paper
- `../presentation/figures/<name>.png` — consumed by the PPT deck builder

To regenerate, run from the A3 repo root (the scripts read from
`results/v*/submission.csv`, which must be present):

```bash
python overleaf-report/figure-sources/build_lb_progression.py
python overleaf-report/figure-sources/build_arch_diagram.py
python overleaf-report/figure-sources/build_pseudo_pipeline.py
python overleaf-report/figure-sources/build_teacher_prob_histogram.py
```

Then re-upload the changed PDFs to Overleaf (drag-replace works).

## Manuscript provenance

Content draws from the project's master outline (`notes/REPORT_OUTLINE.md`)
and the long-form markdown drafts (`notes/06_results.md`,
`notes/07_negative_results.md`), updated for the v47 per-seed finding
(seed 2 at LB 0.8355; the within-recipe ensemble net-negative under an
outlier seed is documented in §VII-G).
