"""Render A3_cancer_challenge.pdf to one PNG per page at 150 DPI."""
from pathlib import Path
import fitz  # PyMuPDF

HERE = Path(__file__).parent.resolve()
PDF = HERE / "A3_cancer_challenge.pdf"
OUT_DIR = HERE / "slides_png"
OUT_DIR.mkdir(exist_ok=True)

# Clean prior renders
for old in OUT_DIR.glob("slide-*.png"):
    old.unlink()

doc = fitz.open(str(PDF))
zoom = 150 / 72  # 150 DPI
mat = fitz.Matrix(zoom, zoom)
for i, page in enumerate(doc, 1):
    pix = page.get_pixmap(matrix=mat, alpha=False)
    out = OUT_DIR / f"slide-{i:02d}.png"
    pix.save(str(out))
    print(f"  -> {out.name}  ({pix.width}x{pix.height})")
doc.close()
print(f"Done: {len(list(OUT_DIR.glob('slide-*.png')))} pages saved to {OUT_DIR}")
