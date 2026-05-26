# IEEE conference paper — build instructions

LaTeX source for the A3 manuscript. Two source files (`main.tex` + `refs.bib`)
plus figure PDFs read from `../figures/`.

## Easiest path: Overleaf (no local LaTeX install needed)

1. Create a new Overleaf project (blank).
2. Upload `main.tex` and `refs.bib` from this folder.
3. Upload the four figure PDFs from `../figures/`:
   - `arch_diagram.pdf`
   - `lb_progression.pdf`
   - `pseudo_pipeline.pdf`
   - `teacher_prob_histogram.pdf`
4. Set compiler to **pdfLaTeX** (Menu → Settings) and main document to `main.tex`.
5. Click **Recompile**. Overleaf handles the bibtex pass automatically.

## Local build (Windows: MiKTeX; macOS/Linux: TeX Live)

Requirements: `texlive-publishers` (provides `IEEEtran.cls`) and
`texlive-fonts-recommended` (provides T1 fontenc). MiKTeX installs missing
packages on-demand by default.

```
cd report/paper
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

Three pdflatex passes are required: first builds `.aux`, bibtex reads it to
write `.bbl`, second pass embeds the bibliography, third pass resolves any
remaining cross-references and page numbers.

Output: `main.pdf` in the same folder.

## Figure paths

`main.tex` sets `\graphicspath{{../figures/}}`. The figure PDFs are produced by
the `build_*.py` scripts in `../figures/` and committed via the curated-
artifacts policy in `.gitignore`. To regenerate them, run the corresponding
`build_*.py` script with the `results/` artifacts present.

## What's in here

| File | Purpose |
|---|---|
| `main.tex` | The paper (IEEE conference template). ~8 pages compiled. |
| `refs.bib` | Bibliography (BibTeX format, 15 entries). |
| `README.md` | This file. |

The manuscript draws content from the project's master outline
(`../../REPORT_OUTLINE.md`) and the two long-form markdown drafts
(`../06_results.md`, `../07_negative_results.md`), updated for the v47 per-seed
finding (seed 2 at LB 0.8355; the within-recipe ensemble net-negative under an
outlier seed is documented in §VII-G).
