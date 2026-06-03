# References

Canonical citations for the methods and ideas this project builds on. The corresponding PDFs are NOT redistributed in this repository; download them via the linked DOI / arXiv / official source.

## Method papers cited directly in the report

### Pseudo-labels and distillation

- **Bucilă, C., Caruana, R., and Niculescu-Mizil, A. (2006).** *Model compression.* In *Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, pp. 535–541. ACM.
  Used in: the precursor to Hinton 2015. The original "compress an ensemble into a single model" framing that Hinton later extended with temperature.
  DOI: [10.1145/1150402.1150464](https://doi.org/10.1145/1150402.1150464)

- **Hinton, G., Vinyals, O., and Dean, J. (2015).** *Distilling the Knowledge in a Neural Network.* arXiv preprint **arXiv:1503.02531**. Originally presented at the NIPS 2014 Deep Learning Workshop.
  Used in: v46 (soft pseudo / dark knowledge).
  arXiv: [1503.02531](https://arxiv.org/abs/1503.02531)

- **Lee, D.-H. (2013).** *Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks.* In *ICML Workshop on Challenges in Representation Learning*.
  Used in: v44 (hard-pseudo at threshold 0.05 / 0.95).
  Official PDF: [pseudo_label_final.pdf](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf) (also linked from `LB_HISTORY.md` § v44 row).

### Test-time and ensemble methods

- **Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., and Wilson, A. G. (2018).** *Averaging Weights Leads to Wider Optima and Better Generalization.* In *Proceedings of the 34th Conference on Uncertainty in Artificial Intelligence (UAI)*.
  Used in: v43 onward — SWA (Stochastic Weight Averaging) over the last 4 epochs of each seed, with a manual BN refresh pass for the averaged model.
  arXiv: [1803.05407](https://arxiv.org/abs/1803.05407)

- **Li, Y., Wang, N., Shi, J., Liu, J., and Hou, X. (2016).** *Revisiting Batch Normalization For Practical Domain Adaptation.* arXiv preprint **arXiv:1603.04779**.
  Used in: v19 onward — AdaBN, refresh BN running stats on the test set in `train()` mode before inference.
  arXiv: [1603.04779](https://arxiv.org/abs/1603.04779)

### Backbone

- **Tan, M. and Le, Q. V. (2019).** *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* In *Proceedings of the 36th International Conference on Machine Learning (ICML)*, pp. 6105–6114.
  Used in: v17 onward — EfficientNet-B0 dual-branch backbone (timm `efficientnet_b0.ra_in1k`).
  arXiv: [1905.11946](https://arxiv.org/abs/1905.11946)

## Local PDF files (not committed)

The user may have copies of these PDFs at the repo root or in a personal `~/papers/` folder for reading; they are intentionally **not** committed to keep repo size small and to avoid redistributing copyrighted material. Filenames seen locally:

| Local filename | Paper |
|----------------|-------|
| `Distilling the Knowledge in a Neural Network.pdf` | Hinton, Vinyals & Dean 2015 |
| `compression.kdd06.pdf` | Bucilă, Caruana & Niculescu-Mizil 2006 |
| `pseudo_label_final.pdf` | Lee 2013 |

These are gitignored via `/*.pdf` in `.gitignore` (root-level only — figure PDFs under `overleaf-report/` are still version-controlled).
