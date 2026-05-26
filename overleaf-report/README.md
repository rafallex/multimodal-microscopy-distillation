# overleaf-report/ — the Overleaf upload bundle

Everything you need to upload to Overleaf lives in this single folder.
Drag and drop **all 6 files below** (excluding this README, which Overleaf
will ignore) into a fresh Overleaf project and click **Recompile**.

## Folder contents

| File | Role |
|---|---|
| `main.tex` | The paper (IEEE conference template, ~8 pages compiled). **This is the canonical source — edit it here.** |
| `refs.bib` | Bibliography (BibTeX format, 13 cited entries). |
| `arch_diagram.pdf` | Figure 1 (dual EfficientNet-B0 + MIL aux). |
| `lb_progression.pdf` | Figure 3 (public-LB progression chart). |
| `pseudo_pipeline.pdf` | Figure 2 (v44 hard → v46 soft → v47 iterated). |
| `teacher_prob_histogram.pdf` | Figure 4 (v46 teacher probability distribution). |
| `README.md` | This file (skip the upload, or upload — Overleaf ignores it). |

## Overleaf upload (recommended)

1. Create a new Overleaf project (blank).
2. Upload all 6 files (`main.tex` + `refs.bib` + the 4 figure PDFs) into the project root.
3. Menu → Settings: set **Compiler = pdfLaTeX**, **Main document = `main.tex`**.
4. Click **Recompile**. BibTeX runs automatically.

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

## Figure path semantics

`main.tex` sets `\graphicspath{{./}{../report/figures/}}`. The two entries
are dual-purpose:

- `./` — finds the figure PDFs sitting next to `main.tex` (works on both
  Overleaf's flat upload AND a local build from inside `overleaf-report/`).
- `../report/figures/` — fallback for a local build, resolves to the
  canonical figure-PDF location in the repo if the copies in this folder
  ever fall out of sync with the originals.

## Regenerating figures

The figure PDFs in this folder are **copies** of the canonical artifacts in
`../report/figures/`, where the figure-build scripts (`build_*.py`) write
their output. If you regenerate any figure, re-sync the copies with:

```powershell
# from the A3 repo root (Windows PowerShell)
Copy-Item report/figures/*.pdf overleaf-report/ -Force
```

```bash
# from the A3 repo root (macOS/Linux/Git Bash)
cp report/figures/*.pdf overleaf-report/
```

Then re-upload the changed PDFs to Overleaf (drag-replace works).

## Manuscript provenance

Content draws from the project's master outline (`../REPORT_OUTLINE.md`)
and the long-form markdown drafts (`../report/06_results.md`,
`../report/07_negative_results.md`), updated for the v47 per-seed finding
(seed 2 at LB 0.8355; the within-recipe ensemble net-negative under an
outlier seed is documented in §VII-G).
