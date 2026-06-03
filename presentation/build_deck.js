/**
 * A3_cancer_challenge.pptx — 12-slide deck built via pptxgenjs.
 *
 * v46-scoped, SSL-free. Final result: v46 soft-target distillation, public LB 0.8236.
 * Design: "Microscopy Indigo" — section numbering, page indicator, em-dash bullets,
 * stat cards, callout strips. Figures embedded from presentation/figures/.
 *
 * Topic: Multimodal Cancer Cell Classification (Uppsala 1MD042 A3).
 */

const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

// === Palette: "Microscopy Indigo" ===
const NAVY    = "1A3A52";
const TEAL    = "0D9488";
const AMBER   = "D97706";
const EMERALD = "059669";
const CRIMSON = "B91C1C";
const CREAM   = "FAF7F2";
const MUTED   = "64748B";
const WHITE   = "FFFFFF";
const SLATE   = "334155";
const LIGHT   = "E5E7EB";

const N_SLIDES = 12;

// === Layout setup ===
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Rafael Tavares Proença";
pres.title  = "Multimodal Cancer Cell Classification";

const REPO_ROOT = path.resolve(__dirname, "..");
const FIG_DIR   = path.join(REPO_ROOT, "presentation", "figures");
const FIG_ARCH  = path.join(FIG_DIR, "arch_diagram.png");
const FIG_HIST  = path.join(FIG_DIR, "teacher_prob_histogram.png");
const FIG_LB    = path.join(FIG_DIR, "lb_progression.png");

// =========================================================================
// === Reusable elements ===
// =========================================================================
function sectionHeader(slide, num, title, page) {
  slide.addText(`§${num}  ·  ${title.toUpperCase()}`, {
    x: 0.30, y: 0.22, w: 7.5, h: 0.30,
    fontFace: "Calibri", fontSize: 10.5, bold: true,
    color: TEAL, charSpacing: 5, margin: 0,
  });
  slide.addText(`${String(page).padStart(2, "0")} / ${N_SLIDES}`, {
    x: 8.30, y: 0.22, w: 1.40, h: 0.30,
    fontFace: "Calibri", fontSize: 10, color: MUTED, align: "right", margin: 0,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.30, y: 0.52, w: 9.40, h: 0.02,
    fill: { color: TEAL, transparency: 60 }, line: { color: TEAL, width: 0 },
  });
}

function slideTitle(slide, text, y = 0.65, color = NAVY) {
  slide.addText(text, {
    x: 0.30, y: y, w: 9.40, h: 0.85,
    fontFace: "Georgia", fontSize: 26, bold: true, color: color, margin: 0,
  });
}

function slideDek(slide, text, y = 1.30) {
  slide.addText(text, {
    x: 0.30, y: y, w: 9.40, h: 0.35,
    fontFace: "Georgia", fontSize: 13, italic: true, color: MUTED, margin: 0,
  });
}

function footer(slide, sectionNum) {
  slide.addText("A3  ·  Multimodal Cancer Cell Classification", {
    x: 0.30, y: 5.30, w: 6.5, h: 0.22,
    fontFace: "Calibri", fontSize: 9, italic: true, color: MUTED, margin: 0,
  });
  slide.addText(`Uppsala 1MD042  ·  §${sectionNum}`, {
    x: 7.00, y: 5.30, w: 2.70, h: 0.22,
    fontFace: "Calibri", fontSize: 9, italic: true, color: MUTED, align: "right", margin: 0,
  });
}

function statCard(slide, x, y, w, h, big, small, bigColor = NAVY, opts = {}) {
  const stripColor = opts.stripColor || bigColor;
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: WHITE }, line: { color: LIGHT, width: 0.5 },
    shadow: { type: "outer", color: "000000", opacity: 0.06, blur: 5, offset: 1.5, angle: 90 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.07, h, fill: { color: stripColor }, line: { color: stripColor, width: 0 },
  });
  slide.addText(big, {
    x: x + 0.20, y: y + 0.10, w: w - 0.30, h: h * 0.55,
    fontFace: "Georgia", fontSize: opts.bigSize || 26, bold: true,
    color: bigColor, align: "left", valign: "top", margin: 0,
  });
  if (small) {
    slide.addText(small, {
      x: x + 0.20, y: y + h * 0.55, w: w - 0.30, h: h * 0.45,
      fontFace: "Calibri", fontSize: opts.smallSize || 10, color: MUTED,
      align: "left", valign: "top", margin: 0,
    });
  }
}

function bulletList(slide, x, y, w, h, items, opts = {}) {
  const color  = opts.color || SLATE;
  const accent = opts.accent || TEAL;
  const size   = opts.size || 12;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: x, y: y, w: 0.04, h: h,
    fill: { color: accent, transparency: 30 }, line: { color: accent, width: 0 },
  });
  const runs = [];
  items.forEach((item) => {
    const text = typeof item === "string" ? item : item.text || item.head;
    const sub  = typeof item === "string" ? null  : item.body || item.sub;
    runs.push({ text: "— ", options: { fontFace: "Calibri", fontSize: size, bold: true, color: accent } });
    runs.push({ text: text, options: { fontFace: "Calibri", fontSize: size, bold: !!sub, color: color, breakLine: !sub } });
    if (sub) {
      runs.push({ text: "  " + sub, options: { fontFace: "Calibri", fontSize: size - 1, color: MUTED, italic: true, breakLine: true } });
    }
  });
  slide.addText(runs, {
    x: x + 0.18, y: y + 0.02, w: w - 0.22, h: h,
    fontFace: "Calibri", fontSize: size, color: color, paraSpaceAfter: 6, valign: "top", margin: 0,
  });
}

function calloutQuote(slide, x, y, w, h, eyebrowText, quote, opts = {}) {
  const bg = opts.bg || NAVY, fg = opts.fg || CREAM, accent = opts.accent || AMBER;
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: bg }, line: { color: bg, width: 0 } });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.07, h, fill: { color: accent }, line: { color: accent, width: 0 } });
  slide.addText(eyebrowText.toUpperCase(), {
    x: x + 0.22, y: y + 0.12, w: w - 0.30, h: 0.30,
    fontFace: "Calibri", fontSize: 9.5, bold: true, color: accent, charSpacing: 4, margin: 0,
  });
  slide.addText(quote, {
    x: x + 0.22, y: y + 0.42, w: w - 0.30, h: h - 0.50,
    fontFace: "Georgia", fontSize: opts.quoteSize || 13, italic: true, color: fg, margin: 0, valign: "top",
  });
}

function calloutStrip(slide, x, y, w, eyebrowText, quote, opts = {}) {
  const bg = opts.bg || NAVY, fg = opts.fg || CREAM, accent = opts.accent || AMBER, h = opts.h || 0.36;
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: bg }, line: { color: bg, width: 0 } });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.07, h, fill: { color: accent }, line: { color: accent, width: 0 } });
  slide.addText([
    { text: eyebrowText.toUpperCase() + "    ", options: { fontFace: "Calibri", fontSize: 9.5, bold: true, color: accent, charSpacing: 3 } },
    { text: quote, options: { fontFace: "Georgia", fontSize: opts.quoteSize || 11, italic: true, color: fg } },
  ], { x: x + 0.24, y: y, w: w - 0.40, h: h, valign: "middle", margin: 0 });
}

function ruleCallout(slide, x, y, w, h, text) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: CREAM }, line: { color: AMBER, width: 1 } });
  slide.addText("RULE  →", {
    x: x + 0.08, y: y + 0.08, w: 1.10, h: h - 0.16,
    fontFace: "Calibri", fontSize: 9.5, bold: true, color: AMBER, charSpacing: 3, valign: "middle", margin: 0,
  });
  slide.addText(text, {
    x: x + 1.15, y: y + 0.08, w: w - 1.25, h: h - 0.16,
    fontFace: "Calibri", fontSize: 10, color: SLATE, valign: "middle", margin: 0,
  });
}

// =========================================================================
// === SLIDE 1: TITLE ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: 5.625, fill: { color: TEAL }, line: { color: TEAL, width: 0 } });
  s.addText("MULTIMODAL CANCER CLASSIFICATION  ·  2026", {
    x: 4.0, y: 0.30, w: 5.7, h: 0.30, fontFace: "Calibri", fontSize: 10.5, bold: true, color: TEAL, charSpacing: 5, align: "right", margin: 0 });
  s.addText("A3  ·  KAGGLE CHALLENGE", {
    x: 0.80, y: 0.30, w: 4.5, h: 0.30, fontFace: "Calibri", fontSize: 10.5, bold: true, color: TEAL, charSpacing: 5, margin: 0 });
  s.addText("Multimodal Cancer Cell\nClassification", {
    x: 0.80, y: 0.95, w: 9.0, h: 1.55, fontFace: "Georgia", fontSize: 38, bold: true, color: WHITE, margin: 0 });
  s.addText("A data-bound problem solved by distillation, not by bigger models", {
    x: 0.80, y: 2.60, w: 9.0, h: 0.40, fontFace: "Georgia", fontSize: 16, italic: true, color: CREAM, margin: 0 });
  s.addText(
    "With 12 training patients and a patient-disjoint test set, the gains came from growing the effective training set — and from reading each failure faithfully.",
    { x: 0.80, y: 3.10, w: 5.5, h: 0.95, fontFace: "Calibri", fontSize: 12, color: CREAM, margin: 0 });

  // Right-side stat strip
  const stripY = 3.05;
  s.addText("BEST PUBLIC LB", {
    x: 6.50, y: stripY, w: 3.20, h: 0.25, fontFace: "Calibri", fontSize: 9.5, bold: true, color: TEAL, charSpacing: 4, margin: 0 });
  s.addText("0.8236", {
    x: 6.50, y: stripY + 0.22, w: 3.20, h: 0.70, fontFace: "Georgia", fontSize: 36, bold: true, color: AMBER, margin: 0 });
  s.addText("v46 · soft-target distillation", {
    x: 6.50, y: stripY + 0.95, w: 3.20, h: 0.25, fontFace: "Calibri", fontSize: 9.5, italic: true, color: CREAM, margin: 0 });
  s.addText("LIFT VS v19", {
    x: 6.50, y: 4.30, w: 1.55, h: 0.22, fontFace: "Calibri", fontSize: 9, bold: true, color: TEAL, charSpacing: 3, margin: 0 });
  s.addText("+0.078", {
    x: 6.50, y: 4.50, w: 1.55, h: 0.45, fontFace: "Georgia", fontSize: 20, bold: true, color: WHITE, margin: 0 });
  s.addText("TRAIN PATIENTS", {
    x: 8.10, y: 4.30, w: 1.60, h: 0.22, fontFace: "Calibri", fontSize: 9, bold: true, color: TEAL, charSpacing: 3, margin: 0 });
  s.addText("12", {
    x: 8.10, y: 4.50, w: 1.55, h: 0.45, fontFace: "Georgia", fontSize: 20, bold: true, color: WHITE, margin: 0 });

  s.addText("Rafael Tavares Proença", {
    x: 0.80, y: 4.62, w: 5.5, h: 0.28, fontFace: "Calibri", fontSize: 13, bold: true, color: WHITE, margin: 0 });
  s.addText("Advanced Deep Learning for Image Processing  ·  1MD042", {
    x: 0.80, y: 4.90, w: 5.5, h: 0.22, fontFace: "Calibri", fontSize: 10.5, color: CREAM, margin: 0 });
  s.addText("Uppsala University  ·  June 4, 2026", {
    x: 0.80, y: 5.12, w: 5.5, h: 0.22, fontFace: "Calibri", fontSize: 10, italic: true, color: MUTED, margin: 0 });

  s.addNotes(
    "Hi — I'm Rafael. Ten minutes, I'll use about seven, on my run at the Multimodal Cancer Cell " +
    "Classification challenge. My final result is v46 at public LB 0.8236 — the mean of three seeds — " +
    "plus 0.078 over the supervised baseline.\n\n" +
    "Grading is on methodology, not the score — so I'll focus on the reasoning: why this is a " +
    "data-bound problem, and how the wins came from growing the labeled set, not from bigger models."
  );
}

// =========================================================================
// === SLIDE 2: §1 — THE CHALLENGE ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 1, "The Challenge", 2);
  slideTitle(s, "Binary cell classification on paired BF + FL");
  slideDek(s, "Two structural facts shape every method choice that follows.");

  statCard(s, 0.30, 1.85, 2.30, 1.10, "114,302", "TRAIN CELLS  ·  12 patients", NAVY, { bigSize: 24, smallSize: 9.5 });
  statCard(s, 2.70, 1.85, 2.30, 1.10, "59,040", "TEST CELLS  ·  patient-disjoint", NAVY, { bigSize: 24, smallSize: 9.5 });
  statCard(s, 5.10, 1.85, 2.30, 1.10, "38.8%", "POS RATE  ·  pos_weight ≈ 1.58", NAVY, { bigSize: 24, smallSize: 9.5 });
  statCard(s, 7.50, 1.85, 2.20, 1.10, "0.8236", "BEST LB  ·  v46 distillation", AMBER, { bigSize: 24, smallSize: 9.5, stripColor: AMBER });

  bulletList(s, 0.30, 3.15, 6.00, 1.60, [
    "Per-cell prediction: is the cell from a cancer patient? (weak labels — every cell inherits its patient's diagnosis)",
    "Paired inputs: brightfield + fluorescence, 128×128 grayscale",
    "12 training patients · ~10k cells each — small N",
    "Strict patient-disjoint test set — OOD generalisation is the dominant failure mode",
    "Metric: AUC · 4 submissions/day · public/private LB split",
  ], { size: 11, accent: TEAL });

  calloutQuote(s, 6.50, 3.15, 3.20, 1.60,
    "Grading basis (instructor)",
    "“What is done and how that is presented” — methodology over leaderboard score.",
    { quoteSize: 11.5 });

  footer(s, 1);
  s.addNotes(
    "The setup. Per-cell binary classification — is this cell from a cancer patient — on paired " +
    "brightfield and fluorescence microscopy, both 128 by 128 grayscale. Labels are patient-level: " +
    "every cell inherits its patient's diagnosis.\n\n" +
    "Two structural facts. First: only 12 training patients — small N. Second, and this dominates " +
    "everything: the test patients are strictly disjoint. Any model that learns 'this is patient 5' " +
    "instead of 'this is cancer' fails on test. The instructor was explicit that grading is on " +
    "methodology, not the leaderboard."
  );
}

// =========================================================================
// === SLIDE 3: §2 — BACKBONE SWEEP ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 2, "Backbone Selection", 3);
  slideTitle(s, "First, fix the backbone — and bigger is worse");
  slideDek(s, "Same recipe, swap the network, fixed 128 px input.");

  // Comparison rows
  const bb = [
    { name: "ResNet-50",        params: "25.6 M", lb: "0.7155", color: MUTED,    note: "5× bigger — scored lower" },
    { name: "EfficientNet-B0",  params: "5.3 M",  lb: "0.7455", color: EMERALD,  note: "winner · the supervised floor (v19)" },
  ];
  let by = 1.95;
  bb.forEach((r) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.30, y: by, w: 6.10, h: 0.86,
      fill: { color: WHITE }, line: { color: LIGHT, width: 0.5 } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.30, y: by, w: 0.07, h: 0.86, fill: { color: r.color }, line: { color: r.color, width: 0 } });
    s.addText(r.name, { x: 0.55, y: by + 0.10, w: 2.7, h: 0.35, fontFace: "Georgia", fontSize: 15, bold: true, color: NAVY, margin: 0 });
    s.addText(r.params + " params", { x: 0.55, y: by + 0.48, w: 2.7, h: 0.28, fontFace: "Calibri", fontSize: 10, color: MUTED, margin: 0 });
    s.addText(r.lb, { x: 3.35, y: by + 0.18, w: 1.30, h: 0.50, fontFace: "Georgia", fontSize: 22, bold: true, color: r.color, margin: 0 });
    s.addText(r.note, { x: 4.70, y: by + 0.20, w: 1.65, h: 0.50, fontFace: "Calibri", fontSize: 9.5, italic: true, color: MUTED, valign: "middle", margin: 0 });
    by += 0.98;
  });
  s.addText("Also prototyped: DenseNet-201, SE-ResNet-50 — no gain to justify the extra capacity.", {
    x: 0.30, y: by + 0.02, w: 6.10, h: 0.30, fontFace: "Calibri", fontSize: 9.5, italic: true, color: MUTED, margin: 0 });

  calloutQuote(s, 6.65, 1.95, 3.05, 1.95,
    "Why the lightest wins",
    "B0 is the smallest EfficientNet (5.3 M). On 12 patients a bigger net has more room to memorise patient identity — exactly the OOD failure mode. Capacity is not the lever; data is.",
    { quoteSize: 11, accent: TEAL });

  footer(s, 2);
  s.addNotes(
    "Before tuning anything I fixed the backbone — same recipe, swap the network, fixed 128 pixels.\n\n" +
    "EfficientNet-B0, 5.3 million parameters, beat ResNet-50 at 25.6 million — 0.7455 vs 0.7155. The " +
    "5× bigger model scored lower. I also prototyped DenseNet-201 and SE-ResNet-50, no gain.\n\n" +
    "The signal that matters: bigger was worse. On 12 patients extra capacity just memorises patient " +
    "identity — the OOD failure mode. So the lightest principled backbone is the right bias. B0 became " +
    "v19, the supervised floor."
  );
}

// =========================================================================
// === SLIDE 4: §3 — THE RECIPE (Fig 1) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 3, "The Recipe", 4);
  slideTitle(s, "Dual EfficientNet-B0 — model the test distribution");
  slideDek(s, "The interventions that matter sit outside the backbone.");

  if (fs.existsSync(FIG_ARCH)) {
    s.addImage({ path: FIG_ARCH, x: 0.30, y: 1.60, w: 6.60, h: 2.55,
      altText: "Dual EfficientNet-B0 backbone, brightfield + fluorescence inputs, late concat fusion, MLP head, per-patient MIL auxiliary loss" });
    s.addText("Fig. 1  Backbone is unremarkable on purpose — the work is at training time (MIL) and test time (AdaBN, stain norm).", {
      x: 0.30, y: 4.20, w: 6.60, h: 0.35, fontFace: "Calibri", fontSize: 8.5, italic: true, color: MUTED, margin: 0 });
  }

  bulletList(s, 7.05, 1.60, 2.65, 3.10, [
    { head: "Late concat fusion", body: "feature-level, dual branch" },
    { head: "Per-patient MIL aux", body: "mean cell logits per patient, BCE, w = 0.5" },
    { head: "AdaBN", body: "refresh BN stats on the test set (Li 2016)" },
    { head: "Test-set stain norm", body: "FL test ≈ 20 % brighter than train" },
    { head: "40-way TTA", body: "scales × D4, prob-averaged" },
  ], { size: 10.5, accent: TEAL });

  footer(s, 3);
  s.addNotes(
    "The recipe — v19, the supervised floor at 0.7455. Two EfficientNet-B0 branches, one per modality, " +
    "late concat, MLP head, via timm with ImageNet weights.\n\n" +
    "The interesting parts are outside the backbone, and they all model the OOD problem. Per-patient " +
    "MIL auxiliary loss regularises against memorising a patient. AdaBN refreshes BatchNorm stats on " +
    "the test set. Test-set stain normalisation — I normalise each modality with TEST statistics, not " +
    "train, because the fluorescence test images are about 20 percent brighter. And 40-way test-time " +
    "augmentation. The theme: import invariance instead of hoping the model learns it."
  );
}

// =========================================================================
// === SLIDE 5: §4 — THE PATH ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 4, "The Path", 5);
  slideTitle(s, "Floor → regularisers → pseudo-labels → distillation");
  slideDek(s, "Each step answered a question. The biggest jump was the last.");

  const TY = 3.00, TX = 0.80, TW = 8.40;
  s.addShape(pres.shapes.RECTANGLE, { x: TX, y: TY, w: TW, h: 0.04, fill: { color: MUTED, transparency: 50 }, line: { color: MUTED, width: 0 } });

  const steps = [
    { px: 1.20, ver: "v19", label: "Supervised floor",  color: NAVY,    result: "0.7455", note: "the baseline" },
    { px: 3.70, ver: "v41", label: "L4 regularisers",   color: NAVY,    result: "0.7563", note: "+0.011" },
    { px: 6.20, ver: "v44", label: "Hard pseudo-labels",color: EMERALD, result: "0.7812", note: "+0.025 · Lee 2013" },
    { px: 8.70, ver: "v46", label: "Soft distillation", color: AMBER,   result: "0.8236", note: "+0.042 · Hinton 2015" },
  ];
  steps.forEach(({ px, ver, label, color, result, note }) => {
    s.addShape(pres.shapes.OVAL, { x: px - 0.16, y: TY - 0.16, w: 0.36, h: 0.36, fill: { color }, line: { color: WHITE, width: 1.5 } });
    s.addText(ver, { x: px - 0.95, y: 1.95, w: 1.90, h: 0.32, fontFace: "Georgia", fontSize: 15, bold: true, color: NAVY, align: "center", margin: 0 });
    s.addText(label, { x: px - 1.00, y: 2.32, w: 2.00, h: 0.45, fontFace: "Calibri", fontSize: 10, italic: true, color: MUTED, align: "center", margin: 0 });
    s.addText(result, { x: px - 0.95, y: 3.45, w: 1.90, h: 0.35, fontFace: "Georgia", fontSize: 16, bold: true, color, align: "center", margin: 0 });
    s.addText(note, { x: px - 1.05, y: 3.82, w: 2.10, h: 0.30, fontFace: "Calibri", fontSize: 9, italic: true, color: MUTED, align: "center", margin: 0 });
  });

  calloutStrip(s, 0.30, 4.55, 9.40,
    "Headline",
    "+0.078 LB from v19 to v46 — more than half of it from the single soft-distillation step.",
    { h: 0.42, quoteSize: 11 });

  footer(s, 4);
  s.addNotes(
    "Four steps. v19 sets the supervised floor at 0.7455. v41 stacks four textbook regularisers " +
    "cleanly for +0.011. v44 adds Lee-2013 hard pseudo-labels for +0.025 — the first sign the " +
    "bottleneck is labeled-set size. And v46 — soft-target distillation — is +0.042, the breakthrough " +
    "and my final result at 0.8236. More than half the total gain is that one step."
  );
}

// =========================================================================
// === SLIDE 6: §5 — ONE CHANGE AT A TIME (v41 vs v43) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 5, "Discipline", 6);
  slideTitle(s, "Change one thing, not four");

  // v41 panel
  s.addShape(pres.shapes.RECTANGLE, { x: 0.30, y: 1.60, w: 4.55, h: 2.85, fill: { color: WHITE }, line: { color: LIGHT, width: 0.7 },
    shadow: { type: "outer", color: "000000", opacity: 0.05, blur: 5, offset: 1.5, angle: 90 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.30, y: 1.60, w: 4.55, h: 0.08, fill: { color: EMERALD }, line: { color: EMERALD, width: 0 } });
  s.addText("v41  ·  clean stack on v19", { x: 0.45, y: 1.75, w: 4.30, h: 0.30, fontFace: "Calibri", fontSize: 10, bold: true, color: TEAL, charSpacing: 2, margin: 0 });
  s.addText("0.7563", { x: 0.45, y: 2.05, w: 2.20, h: 0.55, fontFace: "Georgia", fontSize: 30, bold: true, color: NAVY, margin: 0 });
  s.addText("+0.011", { x: 2.45, y: 2.20, w: 2.30, h: 0.40, fontFace: "Georgia", fontSize: 16, bold: true, color: EMERALD, margin: 0 });
  bulletList(s, 0.45, 2.70, 4.30, 1.70, [
    { head: "label smoothing 0.05", body: "L4 standard" },
    { head: "dropout 0.3 → 0.4", body: "classifier head" },
    { head: "paired RandomResizedCrop", body: "scale 0.85–1.0" },
    { head: "multiscale TTA", body: "3 scales × 8 D4" },
  ], { size: 9.5, accent: EMERALD });

  // v43 panel
  s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y: 1.60, w: 4.55, h: 2.85, fill: { color: WHITE }, line: { color: LIGHT, width: 0.7 },
    shadow: { type: "outer", color: "000000", opacity: 0.05, blur: 5, offset: 1.5, angle: 90 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y: 1.60, w: 4.55, h: 0.08, fill: { color: CRIMSON }, line: { color: CRIMSON, width: 0 } });
  s.addText("v43  ·  four MORE, all at once", { x: 5.30, y: 1.75, w: 4.30, h: 0.30, fontFace: "Calibri", fontSize: 10, bold: true, color: CRIMSON, charSpacing: 2, margin: 0 });
  s.addText("0.7444", { x: 5.30, y: 2.05, w: 2.20, h: 0.55, fontFace: "Georgia", fontSize: 30, bold: true, color: NAVY, margin: 0 });
  s.addText("−0.012", { x: 7.30, y: 2.20, w: 2.30, h: 0.40, fontFace: "Georgia", fontSize: 16, bold: true, color: CRIMSON, margin: 0 });
  bulletList(s, 5.30, 2.70, 4.30, 1.70, [
    { head: "FL-tuned aug", body: "ColorJitter + RandomGamma on FL" },
    { head: "WD bump", body: "1e-4 → 3e-4" },
    { head: "3-seed × SWA", body: "Izmailov 2018, last 4 epochs" },
    { head: "40-way TTA", body: "5 scales × 8 D4" },
  ], { size: 9.5, accent: CRIMSON });

  ruleCallout(s, 0.30, 4.65, 9.40, 0.55,
    "One or two related variables per submitted version — otherwise a regression has no attribution.");

  footer(s, 5);
  s.addNotes(
    "Two contrasting experiments. v41: four well-motivated regularisers, each defensible, roughly " +
    "additive — plus 0.011. v43: four MORE changes at once — and it regressed 0.012, with no way to " +
    "tell which one hurt. Five hours of GPU, no isolable signal. The rule: change one or two related " +
    "things per submission."
  );
}

// =========================================================================
// === SLIDE 7: §6 — THE BREAKTHROUGH (Fig 3 histogram) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 6, "The Breakthrough", 7);
  slideTitle(s, "Hard pseudo drops 84% of cells; soft keeps all");
  slideDek(s, "Same teacher, same predictions — a different way of using them.");

  // table
  s.addText("METHOD", { x: 0.30, y: 1.70, w: 1.90, h: 0.25, fontFace: "Calibri", fontSize: 9, bold: true, color: TEAL, charSpacing: 3, margin: 0 });
  s.addText("CELLS USED", { x: 2.20, y: 1.70, w: 1.85, h: 0.25, fontFace: "Calibri", fontSize: 9, bold: true, color: TEAL, charSpacing: 3, margin: 0 });
  s.addText("Δ LB", { x: 4.05, y: 1.70, w: 0.85, h: 0.25, fontFace: "Calibri", fontSize: 9, bold: true, color: TEAL, charSpacing: 3, margin: 0 });
  const rows = [
    { method: "baseline v41", cells: "114,302",          delta: "—",      color: MUTED },
    { method: "v44 · hard",   cells: "+ 9,350  (16 %)",   delta: "+0.025", color: EMERALD },
    { method: "v46 · soft",   cells: "+ 59,040 (100 %)",  delta: "+0.042", color: AMBER },
  ];
  let ry = 1.98;
  rows.forEach((r) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.30, y: ry, w: 4.60, h: 0.04, fill: { color: LIGHT }, line: { color: LIGHT, width: 0 } });
    s.addText(r.method, { x: 0.30, y: ry + 0.06, w: 1.90, h: 0.30, fontFace: "Calibri", fontSize: 10.5, bold: true, color: NAVY, margin: 0 });
    s.addText(r.cells,  { x: 2.20, y: ry + 0.06, w: 1.85, h: 0.30, fontFace: "Calibri", fontSize: 10.5, color: MUTED, margin: 0 });
    s.addText(r.delta,  { x: 4.05, y: ry + 0.06, w: 0.85, h: 0.30, fontFace: "Georgia", fontSize: 12, bold: true, color: r.color, margin: 0 });
    ry += 0.46;
  });

  calloutQuote(s, 0.30, 3.55, 4.60, 1.35,
    "Dark knowledge (Hinton 2015)",
    "A cell the teacher labels p = 0.78 carries direction + magnitude that hard 0/1 binarisation throws away. Soft targets keep all 59k cells, each weighted by teacher confidence.",
    { quoteSize: 10.5, accent: AMBER });

  if (fs.existsSync(FIG_HIST)) {
    s.addImage({ path: FIG_HIST, x: 5.05, y: 1.65, w: 4.65, h: 2.55,
      altText: "Distribution of v46 teacher predictions over 59,040 test cells; red dashed lines at 0.05 and 0.95 mark the hard-pseudo thresholds; the middle 84% is the soft-only dark-knowledge zone" });
    s.addText("Fig. 2  Teacher predictions over all test cells. Red dashed = hard thresholds; soft keeps every column.", {
      x: 5.05, y: 4.22, w: 4.65, h: 0.35, fontFace: "Calibri", fontSize: 8.5, italic: true, color: MUTED, margin: 0 });
  }

  footer(s, 6);
  s.addNotes(
    "The breakthrough. On 12 patients the bottleneck isn't capacity, it's labeled-set size — so use " +
    "the test set as training signal.\n\n" +
    "v44, Lee 2013 hard pseudo-labels: keep cells the teacher is confident about, above 0.95 or below " +
    "0.05 — about 9 thousand — add them with hard labels. Plus 0.025.\n\n" +
    "v46, Hinton 2015 soft distillation: keep ALL 59 thousand cells, target is the teacher's raw " +
    "probability. The histogram shows it — hard pseudo keeps only the tails; the middle 84 percent is " +
    "thrown away. That middle is the dark knowledge — a cell at 0.78 tells the student more than a hard " +
    "1.0. Six times more cells, each weighted by confidence. Plus 0.042 — my final result, 0.8236."
  );
}

// =========================================================================
// === SLIDE 8: §7 — EVIDENCE (Fig 4 LB chart) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 7, "Evidence", 8);
  slideTitle(s, "The whole story in one chart");

  if (fs.existsSync(FIG_LB)) {
    s.addImage({ path: FIG_LB, x: 0.30, y: 1.55, w: 9.40, h: 3.45,
      altText: "Public LB progression by version: v19 floor 0.7455, v41 regularisers, v44 hard pseudo, v46 soft distillation 0.8236" });
    s.addText("Fig. 3  Public LB by version. v19 floor → v46 distillation = +0.078. Half the gain is the v44→v46 step alone.", {
      x: 0.30, y: 5.02, w: 9.40, h: 0.30, fontFace: "Calibri", fontSize: 9, italic: true, color: MUTED, margin: 0 });
  }
  footer(s, 7);
  s.addNotes(
    "The whole story in one chart. X-axis: version order. Y-axis: public LB. The floor at 0.7455, the " +
    "regulariser win, then the climb through hard and soft pseudo-labels to v46 at 0.8236. The headline " +
    "is plus 0.078 from v19 to v46, and more than half of that is the single soft-distillation step."
  );
}

// =========================================================================
// === SLIDE 9: §8 — WHAT DIDN'T WORK ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 8, "What Didn't Work", 9);
  slideTitle(s, "The failures shaped the recipe");
  slideDek(s, "Two that taught me the most — each with a one-line rule.");

  // two failure cards
  const cards = [
    { x: 0.30, head: "Stacking 4 changes at once (v43)", diag: "−0.012 LB, and no way to tell which change hurt.", rule: "One or two related variables per experiment." },
    { x: 5.05, head: "Ensembling / blending models", diag: "No lift beyond noise. On 12 patients the per-seed spread is ~0.02 AUC.", rule: "Averaging cuts variance, not error — so report the variance-reduced ensemble, not a lucky seed." },
  ];
  cards.forEach((c) => {
    s.addShape(pres.shapes.RECTANGLE, { x: c.x, y: 1.85, w: 4.65, h: 1.75, fill: { color: WHITE }, line: { color: LIGHT, width: 0.7 },
      shadow: { type: "outer", color: "000000", opacity: 0.05, blur: 5, offset: 1.5, angle: 90 } });
    s.addShape(pres.shapes.RECTANGLE, { x: c.x, y: 1.85, w: 4.65, h: 0.08, fill: { color: CRIMSON }, line: { color: CRIMSON, width: 0 } });
    s.addText(c.head, { x: c.x + 0.18, y: 2.00, w: 4.30, h: 0.55, fontFace: "Georgia", fontSize: 13.5, bold: true, color: NAVY, margin: 0 });
    s.addText(c.diag, { x: c.x + 0.18, y: 2.62, w: 4.30, h: 0.85, fontFace: "Calibri", fontSize: 10.5, color: SLATE, margin: 0 });
  });
  ruleCallout(s, 0.30, 3.75, 4.65, 0.70, "One or two related variables per experiment.");
  ruleCallout(s, 5.05, 3.75, 4.65, 0.70, "Trust the noise floor over clever blends.");

  calloutStrip(s, 0.30, 4.65, 9.40, "Take-home",
    "The recipe came from discipline — one-variable experiments and respecting the noise floor.",
    { h: 0.42, quoteSize: 11 });

  footer(s, 8);
  s.addNotes(
    "Two failures that taught me the most. Stacking four changes at once cost 0.012 with no " +
    "attribution — so, one change at a time. And ensembling never beat my best single model: on twelve " +
    "patients the seed spread is wide enough that averaging only reduces variance, it doesn't add signal. " +
    "That's why I report the variance-reduced three-seed ensemble, not a lucky single seed."
  );
}

// =========================================================================
// === SLIDE 10: §9 — METHODS I IMPLEMENTED ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 9, "Methods Implemented", 10);
  slideTitle(s, "Five methods — library where one fits, custom where simple");

  const cols = [
    { h: "METHOD", w: 2.55 }, { h: "PAPER", w: 2.05 }, { h: "HOW I BUILT IT", w: 3.05 }, { h: "IMPACT", w: 1.55 },
  ];
  let cx = 0.30;
  cols.forEach((c) => { s.addText(c.h, { x: cx, y: 1.70, w: c.w, h: 0.25, fontFace: "Calibri", fontSize: 9, bold: true, color: TEAL, charSpacing: 2, margin: 0 }); cx += c.w; });

  const meth = [
    { m: "EfficientNet-B0", p: "Tan & Le 2019", how: "timm, pretrained — dual branch", imp: "backbone", c: MUTED },
    { m: "AdaBN",           p: "Li et al. 2016", how: "custom, ~10 lines from the paper", imp: "OOD", c: MUTED },
    { m: "SWA",             p: "Izmailov 2018",  how: "PyTorch swa_utils", imp: "variance", c: MUTED },
    { m: "Hard pseudo",     p: "Lee 2013",       how: "custom — confident test cells", imp: "+0.025", c: EMERALD },
    { m: "Soft distillation", p: "Hinton 2015",  how: "custom — soft probs, all 59k cells", imp: "+0.042", c: AMBER },
  ];
  let my = 2.00;
  meth.forEach((r) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.30, y: my, w: 9.20, h: 0.04, fill: { color: LIGHT }, line: { color: LIGHT, width: 0 } });
    s.addText(r.m,   { x: 0.30, y: my + 0.05, w: 2.55, h: 0.40, fontFace: "Calibri", fontSize: 11, bold: true, color: NAVY, valign: "middle", margin: 0 });
    s.addText(r.p,   { x: 2.85, y: my + 0.05, w: 2.05, h: 0.40, fontFace: "Calibri", fontSize: 10.5, italic: true, color: MUTED, valign: "middle", margin: 0 });
    s.addText(r.how, { x: 4.90, y: my + 0.05, w: 3.05, h: 0.40, fontFace: "Calibri", fontSize: 10, color: SLATE, valign: "middle", margin: 0 });
    s.addText(r.imp, { x: 7.95, y: my + 0.05, w: 1.55, h: 0.40, fontFace: "Georgia", fontSize: 12, bold: true, color: r.c, valign: "middle", margin: 0 });
    my += 0.52;
  });

  calloutStrip(s, 0.30, 4.70, 9.40, "Honest split",
    "timm for the backbone, PyTorch for SWA — everything that moved the needle I built from the papers.",
    { h: 0.40, quoteSize: 11 });

  footer(s, 9);
  s.addNotes(
    "Every method in the final v46 recipe, with its source and how I built it. EfficientNet-B0 via " +
    "timm. AdaBN I reimplemented in about ten lines from the paper. SWA is PyTorch's swa_utils. The two " +
    "that moved the needle are custom from the papers: Lee-2013 hard pseudo-labels, plus 0.025, and " +
    "Hinton-2015 soft-target distillation, plus 0.042 — my final result. I used a library where a solid " +
    "one existed and implemented the simple ideas directly."
  );
}

// =========================================================================
// === SLIDE 11: §10 — TAKE-HOME ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: 5.625, fill: { color: TEAL }, line: { color: TEAL, width: 0 } });
  s.addText("§10  ·  TAKE-HOME", { x: 0.55, y: 0.35, w: 7.5, h: 0.30, fontFace: "Calibri", fontSize: 10.5, bold: true, color: TEAL, charSpacing: 5, margin: 0 });
  s.addText("11 / " + N_SLIDES, { x: 8.30, y: 0.35, w: 1.40, h: 0.30, fontFace: "Calibri", fontSize: 10, color: CREAM, align: "right", margin: 0 });
  s.addText("Four things I'll carry forward", { x: 0.55, y: 0.80, w: 9.0, h: 0.70, fontFace: "Georgia", fontSize: 26, bold: true, color: WHITE, margin: 0 });

  const takes = [
    { n: "1", h: "On small-N, data > capacity.", b: "Four regularisers gained +0.011; soft distillation gained +0.042 — and a 5× bigger backbone scored worse." },
    { n: "2", h: "Change one thing at a time.", b: "v43 stacked four changes and regressed 0.012 with no attribution." },
    { n: "3", h: "Respect the seed-noise floor.", b: "Same recipe, different seed: ~0.03 LB swing. Single-seed A/B claims under that are noise; ensembling only cuts variance." },
    { n: "4", h: "Model the test distribution.", b: "AdaBN, test-set stain norm and MIL are all responses to the patient-disjoint OOD problem — that's where the recipe came from." },
  ];
  let ty = 1.70;
  takes.forEach((t) => {
    s.addText(t.n, { x: 0.55, y: ty, w: 0.55, h: 0.65, fontFace: "Georgia", fontSize: 30, bold: true, color: AMBER, margin: 0 });
    s.addText([
      { text: t.h + "  ", options: { fontFace: "Calibri", fontSize: 13.5, bold: true, color: WHITE } },
      { text: t.b, options: { fontFace: "Calibri", fontSize: 11.5, color: CREAM, italic: true } },
    ], { x: 1.20, y: ty + 0.02, w: 8.45, h: 0.80, valign: "top", margin: 0 });
    ty += 0.86;
  });

  s.addText("Thank you  ·  questions?", { x: 0.55, y: 5.18, w: 6.0, h: 0.30, fontFace: "Georgia", fontSize: 13, italic: true, color: CREAM, margin: 0 });
  s.addText("github.com/rafallex", { x: 6.50, y: 5.18, w: 3.20, h: 0.30, fontFace: "Calibri", fontSize: 10, color: MUTED, align: "right", margin: 0 });

  s.addNotes(
    "Four take-homes. One: on small-N, data beats capacity — distillation gained four times what " +
    "regularisers did, and the bigger backbone was worse. Two: change one thing at a time. Three: " +
    "respect the seed-noise floor — a seed swing is 0.03, so I trust ensembles for variance reduction, " +
    "not lucky seeds. Four: model the test distribution — every OOD defence is a response to the " +
    "patient-disjoint problem. Thank you — happy to take questions."
  );
}

// =========================================================================
// === SLIDE 12: §11 — REFERENCES ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 11, "References", 12);
  slideTitle(s, "References");
  slideDek(s, "The papers behind every method in the v46 recipe.");

  const refs = [
    "Tan, M. & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML. arXiv:1905.11946",
    "Li, Y., Wang, N., Shi, J., Liu, J. & Hou, X. (2016). Revisiting Batch Normalization for Practical Domain Adaptation. arXiv:1603.04779",
    "Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. & Wilson, A. G. (2018). Averaging Weights Leads to Wider Optima and Better Generalization (SWA). UAI. arXiv:1803.05407",
    "Lee, D.-H. (2013). Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. ICML Workshop.",
    "Hinton, G., Vinyals, O. & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531",
  ];
  const runs = [];
  refs.forEach((r, i) => {
    runs.push({ text: (i + 1) + ".  ", options: { fontFace: "Calibri", fontSize: 12, bold: true, color: TEAL } });
    runs.push({ text: r, options: { fontFace: "Calibri", fontSize: 12, color: SLATE, breakLine: true } });
  });
  s.addText(runs, { x: 0.40, y: 1.85, w: 9.20, h: 2.90, paraSpaceAfter: 12, valign: "top", margin: 0 });

  s.addText("Full citations and the version-by-version log are in REFERENCES.md and LB_HISTORY.md in the repo.", {
    x: 0.40, y: 4.85, w: 9.20, h: 0.30, fontFace: "Calibri", fontSize: 10, italic: true, color: MUTED, margin: 0 });

  footer(s, 11);
  s.addNotes("References — the five papers behind the recipe. Full citations and my version-by-version log are in the repository.");
}

// =========================================================================
const OUT = path.join(REPO_ROOT, "presentation", "A3_cancer_challenge.pptx");
pres.writeFile({ fileName: OUT }).then(() => {
  console.log("Saved " + OUT);
}).catch((e) => { console.error(e); process.exit(1); });
