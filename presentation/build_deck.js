/**
 * A3_cancer_challenge.pptx — 14-slide deck built via pptxgenjs.
 *
 * Design: a custom "Microscopy Indigo" visual language:
 *   - Section numbering §1 ... §10 in the eyebrow on every content slide
 *   - Per-slide page indicator "NN / 12" at top-right
 *   - Footer with course id + section number on every page
 *   - Em-dash "—" bullets aligned to a vertical rule color stripe
 *   - Stat blocks: Georgia bold number (28-32pt) over small Calibri caption
 *   - RULE → callouts on negative-results slide
 *   - Instructor quote callout on §1
 *   - Four figures embedded:
 *       Fig. 1  arch_diagram.png        (dual EffNet-B0 + MIL aux)
 *       Fig. 2  pseudo_pipeline.png     (hard / soft / round-2)
 *       Fig. 3  teacher_prob_histogram.png  (dark knowledge)
 *       Fig. 4  lb_progression.png      (29 versions → v47 #1)
 *
 * Topic: Multimodal Cancer Cell Classification (Uppsala 1MD042 A3)
 * Best public LB: 0.8392 (v58 intermediate co-attention, single model); peaked #1.
 *
 * Palette: "Microscopy Indigo" (content-informed for medical imaging):
 *   NAVY    #1A3A52  deep slate-navy (dark bg, body text on cream)
 *   TEAL    #0D9488  fluorescence cyan (motif accent, section numbers)
 *   AMBER   #D97706  highlight / breakthrough callouts
 *   EMERALD #059669  wins / positive outcomes
 *   CRIMSON #B91C1C  regressions / negative outcomes
 *   CREAM   #FAF7F2  body background
 *   MUTED   #64748B  secondary text, axis labels
 *
 * Typography: Georgia (headers, numbers) + Calibri (body, captions).
 */

const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

// === Palette ===
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

// === Layout setup ===
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";   // 10" × 5.625"
pres.author = "Rafael Tavares Proença";
pres.title  = "Multimodal Cancer Cell Classification";

const REPO_ROOT  = path.resolve(__dirname, "..");
const FIG_DIR    = path.join(REPO_ROOT, "presentation", "figures");
const FIG_ARCH   = path.join(FIG_DIR, "arch_diagram.png");
const FIG_PIPE   = path.join(FIG_DIR, "pseudo_pipeline.png");
const FIG_HIST   = path.join(FIG_DIR, "teacher_prob_histogram.png");
const FIG_LB     = path.join(FIG_DIR, "lb_progression.png");

// =========================================================================
// === Reusable elements ===
// =========================================================================

/** Section eyebrow at top of every content slide.
 *  Format: "§N · SECTION TITLE         NN / 12"
 */
function sectionHeader(slide, num, title, page) {
  slide.addText(`§${num}  ·  ${title.toUpperCase()}`, {
    x: 0.30, y: 0.22, w: 7.5, h: 0.30,
    fontFace: "Calibri", fontSize: 10.5, bold: true,
    color: TEAL, charSpacing: 5, margin: 0,
  });
  slide.addText(`${String(page).padStart(2, "0")} / 14`, {
    x: 8.30, y: 0.22, w: 1.40, h: 0.30,
    fontFace: "Calibri", fontSize: 10, color: MUTED,
    align: "right", margin: 0,
  });
  // Thin teal underline
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.30, y: 0.52, w: 9.40, h: 0.02,
    fill: { color: TEAL, transparency: 60 }, line: { color: TEAL, width: 0 },
  });
}

/** Big slide title in Georgia. */
function slideTitle(slide, text, y = 0.65, color = NAVY) {
  slide.addText(text, {
    x: 0.30, y: y, w: 9.40, h: 0.85,
    fontFace: "Georgia", fontSize: 26, bold: true,
    color: color, margin: 0,
  });
}

/** Optional sub-title / dek line in italic. */
function slideDek(slide, text, y = 1.30) {
  slide.addText(text, {
    x: 0.30, y: y, w: 9.40, h: 0.35,
    fontFace: "Georgia", fontSize: 13, italic: true,
    color: MUTED, margin: 0,
  });
}

/** Footer (course + section indicator). */
function footer(slide, sectionNum, page = null) {
  slide.addText(
    "A3  ·  Multimodal Cancer Cell Classification",
    {
      x: 0.30, y: 5.30, w: 6.5, h: 0.22,
      fontFace: "Calibri", fontSize: 9, italic: true, color: MUTED, margin: 0,
    });
  slide.addText(`Uppsala 1MD042  ·  §${sectionNum}`, {
    x: 7.00, y: 5.30, w: 2.70, h: 0.22,
    fontFace: "Calibri", fontSize: 9, italic: true, color: MUTED,
    align: "right", margin: 0,
  });
}

/** Stat card — big number on top, small caption below.
 *  big = main display string (e.g. "0.8264")
 *  small = small caption ("v47 ensemble LB")
 *  bigColor lets you color-code outcome
 */
function statCard(slide, x, y, w, h, big, small, bigColor = NAVY, opts = {}) {
  const stripColor = opts.stripColor || bigColor;
  // Outer card with subtle shadow
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: WHITE }, line: { color: LIGHT, width: 0.5 },
    shadow: { type: "outer", color: "000000", opacity: 0.06, blur: 5,
              offset: 1.5, angle: 90 },
  });
  // Accent stripe on left
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.07, h,
    fill: { color: stripColor }, line: { color: stripColor, width: 0 },
  });
  // The big number
  slide.addText(big, {
    x: x + 0.20, y: y + 0.10, w: w - 0.30, h: h * 0.55,
    fontFace: "Georgia", fontSize: opts.bigSize || 26, bold: true,
    color: bigColor, align: "left", valign: "top", margin: 0,
  });
  // The caption
  if (small) {
    slide.addText(small, {
      x: x + 0.20, y: y + h * 0.55, w: w - 0.30, h: h * 0.45,
      fontFace: "Calibri", fontSize: opts.smallSize || 10, color: MUTED,
      align: "left", valign: "top", margin: 0,
    });
  }
}

/** Em-dash bullet list with vertical accent stripe.
 *  items = [{ head: "Headline", body: "explanation" }, ...] or strings.
 */
function bulletList(slide, x, y, w, h, items, opts = {}) {
  const color  = opts.color || SLATE;
  const accent = opts.accent || TEAL;
  const size   = opts.size || 12;
  const lineSp = opts.lineSpacing || 1.15;

  // Vertical accent stripe along the left edge
  slide.addShape(pres.shapes.RECTANGLE, {
    x: x, y: y, w: 0.04, h: h,
    fill: { color: accent, transparency: 30 }, line: { color: accent, width: 0 },
  });

  const runs = [];
  items.forEach((item, i) => {
    const text = typeof item === "string" ? item : item.text || item.head;
    const sub  = typeof item === "string" ? null  : item.body || item.sub;
    runs.push({
      text: "— ",
      options: { fontFace: "Calibri", fontSize: size, bold: true, color: accent },
    });
    runs.push({
      text: text,
      options: { fontFace: "Calibri", fontSize: size, bold: !!sub, color: color,
                 breakLine: !sub },
    });
    if (sub) {
      runs.push({
        text: "  " + sub,
        options: { fontFace: "Calibri", fontSize: size - 1, color: MUTED,
                   italic: true, breakLine: true },
      });
    }
  });

  slide.addText(runs, {
    x: x + 0.18, y: y + 0.02, w: w - 0.22, h: h,
    fontFace: "Calibri", fontSize: size, color: color,
    paraSpaceAfter: 6, valign: "top", margin: 0,
  });
}

/** Pull-quote / callout block. */
function calloutQuote(slide, x, y, w, h, eyebrowText, quote, opts = {}) {
  const bg     = opts.bg || NAVY;
  const fg     = opts.fg || CREAM;
  const accent = opts.accent || AMBER;
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: bg }, line: { color: bg, width: 0 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.07, h,
    fill: { color: accent }, line: { color: accent, width: 0 },
  });
  slide.addText(eyebrowText.toUpperCase(), {
    x: x + 0.22, y: y + 0.12, w: w - 0.30, h: 0.30,
    fontFace: "Calibri", fontSize: 9.5, bold: true,
    color: accent, charSpacing: 4, margin: 0,
  });
  slide.addText(quote, {
    x: x + 0.22, y: y + 0.42, w: w - 0.30, h: h - 0.50,
    fontFace: "Georgia", fontSize: opts.quoteSize || 13, italic: true,
    color: fg, margin: 0, valign: "top",
  });
}

/** Single-line callout strip (eyebrow + quote on ONE line) for tight bottom bands.
 *  calloutQuote stacks eyebrow over quote and needs ~0.6"; this fits in ~0.34". */
function calloutStrip(slide, x, y, w, eyebrowText, quote, opts = {}) {
  const bg     = opts.bg || NAVY;
  const fg     = opts.fg || CREAM;
  const accent = opts.accent || AMBER;
  const h      = opts.h || 0.36;
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: bg }, line: { color: bg, width: 0 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.07, h, fill: { color: accent }, line: { color: accent, width: 0 },
  });
  slide.addText([
    { text: eyebrowText.toUpperCase() + "    ",
      options: { fontFace: "Calibri", fontSize: 9.5, bold: true, color: accent, charSpacing: 3 } },
    { text: quote,
      options: { fontFace: "Georgia", fontSize: opts.quoteSize || 11, italic: true, color: fg } },
  ], { x: x + 0.24, y: y, w: w - 0.40, h: h, valign: "middle", margin: 0 });
}

/** RULE → callout for negative-results slide. */
function ruleCallout(slide, x, y, w, h, text) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: CREAM }, line: { color: AMBER, width: 1 },
  });
  // Left arrow icon
  slide.addText("RULE  →", {
    x: x + 0.08, y: y + 0.08, w: 1.10, h: h - 0.16,
    fontFace: "Calibri", fontSize: 9.5, bold: true,
    color: AMBER, charSpacing: 3, valign: "middle", margin: 0,
  });
  slide.addText(text, {
    x: x + 1.15, y: y + 0.08, w: w - 1.25, h: h - 0.16,
    fontFace: "Calibri", fontSize: 10, color: SLATE,
    valign: "middle", margin: 0,
  });
}

// =========================================================================
// === SLIDE 1: TITLE ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // Left accent bar (teal)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.22, h: 5.625,
    fill: { color: TEAL }, line: { color: TEAL, width: 0 },
  });

  // Top-right corner small caps eyebrow
  s.addText("MULTIMODAL CANCER CLASSIFICATION  ·  2026", {
    x: 4.0, y: 0.30, w: 5.7, h: 0.30,
    fontFace: "Calibri", fontSize: 10.5, bold: true,
    color: TEAL, charSpacing: 5, align: "right", margin: 0,
  });

  // Eyebrow (top-left)
  s.addText("A3  ·  KAGGLE CHALLENGE", {
    x: 0.80, y: 0.30, w: 4.5, h: 0.30,
    fontFace: "Calibri", fontSize: 10.5, bold: true,
    color: TEAL, charSpacing: 5, margin: 0,
  });

  // Big title (Georgia, multi-line)
  s.addText("Multimodal Cancer Cell\nClassification", {
    x: 0.80, y: 0.90, w: 9.0, h: 1.55,
    fontFace: "Georgia", fontSize: 38, bold: true, color: WHITE,
    margin: 0,
  });

  // Dek (italic, cream)
  s.addText("30 versions, six diagnosed failures, one breakthrough", {
    x: 0.80, y: 2.55, w: 9.0, h: 0.40,
    fontFace: "Georgia", fontSize: 16, italic: true, color: CREAM,
    margin: 0,
  });

  // Body paragraph
  s.addText(
    "A methodological run at the Multimodal Cancer Classification challenge — where each diagnosed failure shaped the next version, and the wins came from reading the losses faithfully.",
    {
      x: 0.80, y: 3.05, w: 5.5, h: 0.85,
      fontFace: "Calibri", fontSize: 12, color: CREAM,
      margin: 0,
    });

  // Right-side stat strip — three stat blocks for headline numbers
  const stripY = 3.05;
  const stripH = 1.10;
  // PUBLIC LB
  s.addText("PUBLIC LB", {
    x: 6.50, y: stripY, w: 3.20, h: 0.25,
    fontFace: "Calibri", fontSize: 9.5, bold: true,
    color: TEAL, charSpacing: 4, margin: 0,
  });
  s.addText("0.8392", {
    x: 6.50, y: stripY + 0.22, w: 3.20, h: 0.70,
    fontFace: "Georgia", fontSize: 36, bold: true, color: AMBER, margin: 0,
  });
  s.addText("v58 co-attention seed2  ·  #3 on LB", {
    x: 6.50, y: stripY + 0.95, w: 3.20, h: 0.25,
    fontFace: "Calibri", fontSize: 9.5, italic: true, color: CREAM, margin: 0,
  });

  // LIFT VS V19 + VERSIONS row
  s.addText("LIFT VS V19", {
    x: 6.50, y: 4.30, w: 1.55, h: 0.22,
    fontFace: "Calibri", fontSize: 9, bold: true,
    color: TEAL, charSpacing: 3, margin: 0,
  });
  s.addText("+0.094", {
    x: 6.50, y: 4.50, w: 1.55, h: 0.45,
    fontFace: "Georgia", fontSize: 20, bold: true, color: WHITE, margin: 0,
  });

  s.addText("VERSIONS", {
    x: 8.10, y: 4.30, w: 1.55, h: 0.22,
    fontFace: "Calibri", fontSize: 9, bold: true,
    color: TEAL, charSpacing: 3, margin: 0,
  });
  s.addText("30", {
    x: 8.10, y: 4.50, w: 1.55, h: 0.45,
    fontFace: "Georgia", fontSize: 20, bold: true, color: WHITE, margin: 0,
  });

  // Author block (lower-left)
  s.addText("Rafael Tavares Proença", {
    x: 0.80, y: 4.62, w: 5.5, h: 0.28,
    fontFace: "Calibri", fontSize: 13, bold: true, color: WHITE, margin: 0,
  });
  s.addText("Advanced Deep Learning for Image Processing  ·  1MD042", {
    x: 0.80, y: 4.90, w: 5.5, h: 0.22,
    fontFace: "Calibri", fontSize: 10.5, color: CREAM, margin: 0,
  });
  s.addText("Uppsala University  ·  June 4, 2026", {
    x: 0.80, y: 5.12, w: 5.5, h: 0.22,
    fontFace: "Calibri", fontSize: 10, italic: true, color: MUTED, margin: 0,
  });

  // BF / FL small labels (decorative, lower right)
  s.addText("BF  ·  128 × 128", {
    x: 6.50, y: 5.18, w: 1.55, h: 0.20,
    fontFace: "Calibri", fontSize: 8.5, color: MUTED, charSpacing: 2, margin: 0,
  });
  s.addText("FL  ·  128 × 128", {
    x: 8.10, y: 5.18, w: 1.55, h: 0.20,
    fontFace: "Calibri", fontSize: 8.5, color: MUTED, charSpacing: 2, margin: 0,
  });

  s.addNotes(
    "Hello — I'm Rafael. Today I'll walk you through the Multimodal Cancer Cell " +
    "Classification Challenge: 30+ logged Kaggle versions, five acts of experimentation, and " +
    "a final architecture push. +0.094 LB lift over the supervised baseline to my best single " +
    "model: v58's intermediate co-attention fusion (seed2) at 0.8392 -- the new best, holding " +
    "#3 on the public leaderboard.\n\n" +
    "But this talk is not primarily about the score. The instructor said grading is " +
    "based on methodology and presentation, not the leaderboard itself. So I'll spend " +
    "most of the time on what didn't work and why — six diagnosed failures shaped the " +
    "winning recipe directly.\n\n" +
    "Pacing: 14 slides, ~45 seconds per slide, ~30 seconds buffer for questions."
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

  // 4 stat blocks across the top
  statCard(s, 0.30, 1.85, 2.30, 1.10, "114,302",
           "TRAIN CELLS  ·  12 patients", NAVY,
           { bigSize: 24, smallSize: 9.5 });
  statCard(s, 2.70, 1.85, 2.30, 1.10, "59,040",
           "TEST CELLS  ·  patient-disjoint", NAVY,
           { bigSize: 24, smallSize: 9.5 });
  statCard(s, 5.10, 1.85, 2.30, 1.10, "38.8%",
           "POS RATE  ·  pos_weight ≈ 1.58", NAVY,
           { bigSize: 24, smallSize: 9.5 });
  statCard(s, 7.50, 1.85, 2.20, 1.10, "0.8392",
           "BEST LB  ·  v58 co-attention", AMBER,
           { bigSize: 24, smallSize: 9.5, stripColor: AMBER });

  // Bullets
  bulletList(s, 0.30, 3.15, 6.00, 1.60, [
    "Per-cell prediction: is the cell from a cancer patient? (weak labels — MAC hypothesis, all cells inherit patient diagnosis)",
    "Paired inputs: brightfield + fluorescence, 128×128 grayscale",
    "12 training patients · ~10k cells each — small N",
    "Strict patient-disjoint test set — OOD generalisation is the dominant failure mode",
    "Metric: AUC · 4 submissions/day · public/private LB split",
  ], { size: 11, accent: TEAL });

  // Instructor quote callout (lower-right)
  calloutQuote(s, 6.50, 3.15, 3.20, 1.60,
    "Grading basis (instructor, May 14)",
    "“What is done and how that is presented” — methodology over leaderboard score.",
    { quoteSize: 11.5 });

  footer(s, 1);
  s.addNotes(
    "The setup. Per-cell binary classification: is this oral cell from a cancer patient. " +
    "Inputs are paired brightfield and fluorescence microscopy, both 128 by 128 grayscale.\n\n" +
    "Two structural challenges. First: only 12 training patients. Small-N — every label " +
    "is a per-patient grouping. Second: the test set has strict patient-disjoint OOD. That " +
    "second constraint dominates everything that follows.\n\n" +
    "The instructor was explicit on May 14: grading is on methodology and presentation, " +
    "not on the leaderboard score. Peaking at #1 (and sitting top-3 at submission) is " +
    "reassuring, but it's not what's being evaluated. The negative results and the " +
    "diagnostic reasoning behind v46/v47 are."
  );
}

// =========================================================================
// === SLIDE 3: §2 — METHOD (with Fig. 1) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 2, "Method", 3);
  slideTitle(s, "Dual EfficientNet-B0 with patient-MIL aux loss");
  slideDek(s, "The interventions that matter sit outside the backbone.");

  // Embed Fig. 1 (architecture diagram)
  if (fs.existsSync(FIG_ARCH)) {
    s.addImage({
      path: FIG_ARCH,
      x: 0.30, y: 1.55, w: 6.70, h: 2.60,
      altText: "Dual EfficientNet-B0 backbone with brightfield + fluorescence inputs, late concat fusion, MLP head, and per-patient MIL auxiliary loss branch",
    });
    s.addText("Fig. 1  Backbone is unremarkable on purpose. The interesting parts are at training time (MIL) and test time (AdaBN, stain norm).",
      {
        x: 0.30, y: 4.20, w: 6.70, h: 0.35,
        fontFace: "Calibri", fontSize: 8.5, italic: true, color: MUTED, margin: 0,
      });
  }

  // Right-column method bullets
  bulletList(s, 7.10, 1.55, 2.60, 3.20, [
    { head: "Late concat fusion",    body: "feature-level after GAP (v38 early fusion regressed −0.031)" },
    { head: "Per-patient MIL aux",   body: "mean-pool cell logits per patient, BCE, w=0.5" },
    { head: "AdaBN",                 body: "refresh BN stats on test set (Li 2016)" },
    { head: "Test-set stain norm",   body: "closes the 19.7 % FL mean gap" },
    { head: "8-way D4 TTA",          body: "dihedral group, prob-averaged" },
  ], { size: 10.5, accent: TEAL });

  footer(s, 2);
  s.addNotes(
    "Method. Two EfficientNet-B0 backbones, one for brightfield and one for fluorescence, " +
    "each starting from timm's ra_in1k ImageNet pretrain. Grayscale conv stems, ≈4M params each. " +
    "Global average pool from each branch, concatenate the two 1280-d features, MLP head, sigmoid.\n\n" +
    "The interesting parts are NOT the backbone — they're the test-time and training-time " +
    "interventions. (1) Per-patient MIL aux loss: mean cell logits per patient and BCE that " +
    "against the patient label, weighted 0.5. Regularizes against pure within-patient memorization. " +
    "(2) AdaBN: one forward pass over the test set in train mode refreshes BatchNorm running " +
    "stats to the test distribution. (3) Test-set stain normalization: FL test is measurably " +
    "19.7 % brighter than FL train; we use TEST mean/std for normalization. (4) 8-way D4 TTA, " +
    "averaged in probability space.\n\n" +
    "v38 tried early fusion (single 2-channel backbone). Regressed −0.031 LB. Late concat won."
  );
}

// =========================================================================
// === SLIDE 4: §3 — THE ARC (5 acts) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 3, "The Arc", 4);
  slideTitle(s, "30 versions, 5 acts, one breakthrough");
  slideDek(s, "Each act answered a question. Each result changed the next question.");

  // Horizontal timeline with 5 acts
  const TY = 3.05;
  const TX = 0.50;
  const TW = 9.00;
  s.addShape(pres.shapes.RECTANGLE, {
    x: TX, y: TY, w: TW, h: 0.04,
    fill: { color: MUTED, transparency: 50 }, line: { color: MUTED, width: 0 },
  });

  const acts = [
    { px: 1.20, name: "Act 1+2", vers: "v10–v16", label: "Baselines & SSL",
      color: CRIMSON, result: "0.572",   note: "SSL collapse" },
    { px: 3.10, name: "Act 3",    vers: "v17–v19", label: "EffNet pivot",
      color: NAVY,    result: "0.7455",  note: "supervised floor" },
    { px: 5.00, name: "Act 4",    vers: "v20–v41", label: "Regularizer search",
      color: NAVY,    result: "0.7563",  note: "v41 +0.011" },
    { px: 6.90, name: "Act 5A",   vers: "v42–v44", label: "Hard pseudo-labels",
      color: EMERALD, result: "0.7812",  note: "v44 +0.025 (Lee 2013)" },
    { px: 8.90, name: "Act 5B",   vers: "v46→v47", label: "Soft distillation",
      color: AMBER,   result: "0.8264",  note: "+0.039 + 0.003 · final pick" },
  ];

  acts.forEach(({ px, name, vers, label, color, result, note }) => {
    // Dot
    s.addShape(pres.shapes.OVAL, {
      x: px - 0.16, y: TY - 0.16, w: 0.36, h: 0.36,
      fill: { color }, line: { color: WHITE, width: 1.5 },
    });
    // Act name
    s.addText(name.toUpperCase(), {
      x: px - 0.90, y: 1.75, w: 1.80, h: 0.25,
      fontFace: "Calibri", fontSize: 9.5, bold: true,
      color: TEAL, align: "center", charSpacing: 2, margin: 0,
    });
    // Versions
    s.addText(vers, {
      x: px - 0.90, y: 2.05, w: 1.80, h: 0.30,
      fontFace: "Georgia", fontSize: 13, bold: true, color: NAVY,
      align: "center", margin: 0,
    });
    // Label
    s.addText(label, {
      x: px - 0.95, y: 2.40, w: 1.90, h: 0.30,
      fontFace: "Calibri", fontSize: 10, italic: true, color: MUTED,
      align: "center", margin: 0,
    });
    // Result
    s.addText(result, {
      x: px - 0.95, y: 3.50, w: 1.90, h: 0.35,
      fontFace: "Georgia", fontSize: 15, bold: true, color,
      align: "center", margin: 0,
    });
    // Note
    s.addText(note, {
      x: px - 1.00, y: 3.85, w: 2.00, h: 0.30,
      fontFace: "Calibri", fontSize: 9, italic: true, color: MUTED,
      align: "center", margin: 0,
    });
  });

  // Bottom callout
  calloutStrip(s, 0.30, 4.55, 9.40,
    "Headline",
    "+0.094 LB total — the biggest single jump (+0.039) was soft-pseudo distillation (Hinton 2015).",
    { h: 0.42, quoteSize: 11 });

  footer(s, 3);
  s.addNotes(
    "Five acts, anchored to version numbers. We tracked every Kaggle submission in a " +
    "LB_HISTORY.md file in the repo, so when I say 'v19' or 'v47' I mean a specific notebook " +
    "with a specific recipe.\n\n" +
    "Act 1+2: v10-v16, baselines and an ambitious CoMIR-style SSL attempt. v15 scored 0.866 " +
    "in cross-validation — our highest CV ever — then collapsed to 0.572 on the LB.\n\n" +
    "Act 3: v17-v19, the pivot back to a simple EffNet-B0 recipe. LB 0.7455 supervised floor.\n\n" +
    "Act 4: v20-v41, regularizer search. v41 stacked four L4 regularizers cleanly: +0.011.\n\n" +
    "Act 5A: v42-v44, semi-supervised pivot. v42 SSL collapsed; v44 added Lee 2013 hard " +
    "pseudo-labels for +0.025. \n\n" +
    "Act 5B: v46-v47, soft distillation. v46 used Hinton 2015 soft pseudo for +0.039 and " +
    "took #1. v47 iterated with v46 as the new teacher (Xie 2020 noisy student round 2): " +
    "+0.003 on the ensemble (0.8264) but +0.012 on the best single seed (0.8355). We peaked " +
    "#1 here; the noise-floor analysis shows the round-2 ensemble gain is statistically zero, " +
    "and the per-seed dispersion is the real story."
  );
}

// =========================================================================
// === SLIDE 5: §4 — ACT 2 (SSL collapse) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 4, "Act 2 — The Ambition", 5);
  slideTitle(s, "v15 / v16: best CV I ever saw, worst LB");
  slideDek(s, "CoMIR-style cross-modal SSL — and a 29-point generalisation gap.");

  // Left column: what we built (bullets)
  s.addText("What we built", {
    x: 0.30, y: 1.65, w: 5.00, h: 0.28,
    fontFace: "Calibri", fontSize: 10.5, bold: true,
    color: TEAL, charSpacing: 3, margin: 0,
  });
  bulletList(s, 0.30, 1.95, 5.00, 1.80, [
    "Two ResNet-18 branches (BF + FL) with projection heads",
    "NT-Xent contrastive pretrain, τ = 0.1 · 10 epochs",
    "Supervised fine-tune · 8 epochs · discriminative LR",
    { head: "3-fold stratified-group CV", body: "hid the failure — two of three folds had val patients similar to train",
      // mark as the bad bullet
    },
  ], { size: 11, accent: CRIMSON, color: SLATE });

  // Right column: the shock (two stat cards stacked)
  statCard(s, 5.60, 1.65, 4.10, 1.05, "0.866",
           "CV AUC  ·  3-fold stratified-group  ·  best I'd ever seen",
           EMERALD, { bigSize: 32 });
  statCard(s, 5.60, 2.80, 4.10, 1.05, "0.572",
           "PUBLIC LB · v15 submission · 29 AUC points of OOD gap",
           CRIMSON, { bigSize: 32 });
  statCard(s, 5.60, 3.95, 4.10, 0.85,
           "0.943 → 0.503",
           "v16 RESPONSE · 9 careful fixes · LOPO CV — patient-OOF up, LB still random",
           CRIMSON, { bigSize: 20 });

  // Diagnosis bottom strip
  calloutQuote(s, 0.30, 4.05, 5.00, 0.75,
    "Diagnosis",
    "Paired (BF, FL) of cell i always share a patient. Contrastive task solved at ≥95 % by encoding patient identity, not cell content.",
    { quoteSize: 10.5, accent: CRIMSON });

  footer(s, 4);
  s.addNotes(
    "Act 2: CoMIR-style cross-modal contrastive SSL. Two ResNet-18 encoders for brightfield " +
    "and fluorescence, paired images of the same cell as positives, unpaired as negatives. " +
    "Pretrain 10 epochs, fine-tune 8. The CV came in at 0.866 — our highest ever — so we submitted, " +
    "expecting ~0.75. We got 0.572. Essentially random.\n\n" +
    "Diagnosis took us a few days. The issue: every cell has exactly one source patient. Paired " +
    "(BF, FL) of cell i ALWAYS share a patient. Unpaired pairs of i ≠ j USUALLY don't share a " +
    "patient. So the contrastive task can be solved at ≥95 % accuracy by encoding patient identity " +
    "rather than cell content. The pretrain backbone optimized that shortcut. The supervised stage " +
    "memorized within-patient class distributions. Test patients were disjoint. Collapse.\n\n" +
    "v16 was a careful 9-fix response that switched to LOPO CV (12-fold leave-one-patient-out). " +
    "Patient-level OOF went up to 0.943 — best we'd ever seen. But the public LB stayed at " +
    "0.503 — there's a level of OOD that LOPO can't detect. The whole site/protocol differs " +
    "between train and test."
  );
}

// =========================================================================
// === SLIDE 6: §5 — ACT 3 (v19 pivot) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 5, "Act 3 — The Pivot", 6);
  slideTitle(s, "v19: drop the SSL, model the test distribution");
  slideDek(s, "Five ingredients. None of them are new. Together: +0.174 LB over v15.");

  // Three stat blocks across the top
  statCard(s, 0.30, 1.65, 3.10, 1.20, "0.7455",
           "PUBLIC LB · v19 · supervised floor · held ~3 weeks",
           EMERALD, { bigSize: 32 });
  statCard(s, 3.55, 1.65, 3.10, 1.20, "+0.174",
           "VS V15 SSL collapse → floor",
           AMBER, { bigSize: 32 });
  statCard(s, 6.80, 1.65, 2.90, 1.20, "−0.038",
           "VS prior LB leader · 0.7832 at the time",
           NAVY, { bigSize: 32 });

  // Ingredients bullet list
  s.addText("Five ingredients in v19", {
    x: 0.30, y: 3.05, w: 9.40, h: 0.28,
    fontFace: "Calibri", fontSize: 10.5, bold: true,
    color: TEAL, charSpacing: 3, margin: 0,
  });
  bulletList(s, 0.30, 3.35, 9.40, 1.40, [
    "EfficientNet-B0 dual-branch (BF + FL), ImageNet-pretrained, late concat fusion",
    "Per-patient MIL auxiliary loss — patient-mean-logit BCE, weight 0.5",
    "AdaBN — refresh BN running stats on the test set (Li 2016)",
    "Test-set stain normalisation — pixel mean/std from test, not train",
    "Strong paired augmentation + 8-way D4 group TTA, prob-averaged",
  ], { size: 11, accent: EMERALD });

  // Take-home strip
  calloutStrip(s, 0.30, 4.80, 9.40,
    "Take-home",
    "The win wasn't a new architecture — it was treating the test distribution as the thing to model.",
    { h: 0.40, quoteSize: 11 });

  footer(s, 5);
  s.addNotes(
    "The pivot. Drop the SSL complexity entirely. Drop the LOPO CV. Train a standard EffNet-B0 " +
    "with five ingredients that directly model the OOD problem.\n\n" +
    "Five ingredients: dual-branch backbone with late concat fusion. Per-patient MIL aux loss " +
    "regularizing against within-patient memorization. AdaBN to refresh BN stats to the test " +
    "distribution. Test-set stain normalization closing the 19.7 % FL mean gap. Strong " +
    "augmentation plus 8-way dihedral TTA.\n\n" +
    "Public LB 0.7455. Up 0.174 from v15. The new supervised floor that held for three weeks.\n\n" +
    "Take-home: the win came from treating the test distribution as the thing to model. The " +
    "SSL pipeline tried to learn invariance from scratch and failed. v19 imported invariance " +
    "via TTA, AdaBN, and stain norm. Cheaper and more effective."
  );
}

// =========================================================================
// === SLIDE 7: §6 — ACT 4 (v41 vs v43) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 6, "Act 4 — The Search", 7);
  slideTitle(s, "v41 vs v43: change one thing, not four");

  // v41 panel (left)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.30, y: 1.60, w: 4.55, h: 2.85,
    fill: { color: WHITE }, line: { color: LIGHT, width: 0.7 },
    shadow: { type: "outer", color: "000000", opacity: 0.05, blur: 5, offset: 1.5, angle: 90 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.30, y: 1.60, w: 4.55, h: 0.08,
    fill: { color: EMERALD }, line: { color: EMERALD, width: 0 },
  });
  s.addText("v41  ·  clean stack on top of v19", {
    x: 0.45, y: 1.75, w: 4.30, h: 0.30,
    fontFace: "Calibri", fontSize: 10, bold: true, color: TEAL,
    charSpacing: 2, margin: 0,
  });
  s.addText("0.7563", {
    x: 0.45, y: 2.05, w: 2.20, h: 0.55,
    fontFace: "Georgia", fontSize: 30, bold: true, color: NAVY, margin: 0,
  });
  s.addText("+0.011", {
    x: 2.45, y: 2.20, w: 2.30, h: 0.40,
    fontFace: "Georgia", fontSize: 16, bold: true, color: EMERALD, margin: 0,
  });
  bulletList(s, 0.45, 2.70, 4.30, 1.70, [
    { head: "label smoothing ε = 0.05",   body: "L4 standard" },
    { head: "dropout 0.3 → 0.4",           body: "classifier head" },
    { head: "paired RRC",                  body: "RandomResizedCrop scale 0.85-1.0" },
    { head: "multiscale TTA",              body: "3 scales × 8 D4 = 24-way" },
  ], { size: 9.5, accent: EMERALD });

  // v43 panel (right)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.15, y: 1.60, w: 4.55, h: 2.85,
    fill: { color: WHITE }, line: { color: LIGHT, width: 0.7 },
    shadow: { type: "outer", color: "000000", opacity: 0.05, blur: 5, offset: 1.5, angle: 90 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.15, y: 1.60, w: 4.55, h: 0.08,
    fill: { color: CRIMSON }, line: { color: CRIMSON, width: 0 },
  });
  s.addText("v43  ·  four MORE, stacked at once", {
    x: 5.30, y: 1.75, w: 4.30, h: 0.30,
    fontFace: "Calibri", fontSize: 10, bold: true, color: CRIMSON,
    charSpacing: 2, margin: 0,
  });
  s.addText("0.7444", {
    x: 5.30, y: 2.05, w: 2.20, h: 0.55,
    fontFace: "Georgia", fontSize: 30, bold: true, color: NAVY, margin: 0,
  });
  s.addText("−0.012", {
    x: 7.30, y: 2.20, w: 2.30, h: 0.40,
    fontFace: "Georgia", fontSize: 16, bold: true, color: CRIMSON, margin: 0,
  });
  bulletList(s, 5.30, 2.70, 4.30, 1.70, [
    { head: "FL-tuned aug",   body: "CJ 0.5/0.3 + RandomGamma on FL only" },
    { head: "WD bump",         body: "1e-4 → 3e-4" },
    { head: "3-seed × SWA",    body: "Izmailov 2018 · last 4 epochs" },
    { head: "40-way TTA",      body: "5 scales × 8 D4" },
  ], { size: 9.5, accent: CRIMSON });

  // RULE callout
  ruleCallout(s, 0.30, 4.65, 9.40, 0.55,
    "One or two related variables per submitted version. v46 later reverted the FL-tuned aug + WD bump and recovered +0.079.");

  footer(s, 6);
  s.addNotes(
    "Act 4 had two contrasting experiments — one clean, one cautionary.\n\n" +
    "v41 was the clean win. Four well-motivated L4 textbook additions on top of v19: label " +
    "smoothing 0.05, dropout 0.3 → 0.4, paired RandomResizedCrop, multiscale TTA. Each component " +
    "individually defensible, all roughly additive. +0.011 LB to 0.7563.\n\n" +
    "v43 was the cautionary tale. Four MORE changes simultaneously: FL-tuned aug, weight decay " +
    "bump 1e-4 → 3e-4, 3-seed × SWA ensembling, 40-way TTA. Regressed by 0.012. We could NOT " +
    "isolate the bad component. Five hours of Kaggle GPU produced no isolable signal.\n\n" +
    "Rule encoded: change one or two related variables per submitted version. v46 later proved " +
    "this by reverting the FL-tuned aug and the WD bump (keeping the SWA ensemble + extended TTA) " +
    "and gained +0.079 LB. So the FL aug and the WD bump were the bad components."
  );
}

// =========================================================================
// === SLIDE 8: §7 — ACT 5 — THE BREAKTHROUGH (with Fig. 2) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 7, "Act 5 — The Breakthrough", 8);
  slideTitle(s, "Use the test set itself as training signal");

  // Embed Fig. 2 (pseudo pipeline) — full-width
  if (fs.existsSync(FIG_PIPE)) {
    s.addImage({
      path: FIG_PIPE,
      x: 0.30, y: 1.50, w: 9.40, h: 3.20,
      altText: "Pseudo-label pipeline: teacher predicts test cells, hard branch (Lee 2013) keeps 9k cells, soft branch (Hinton 2015) keeps all 59k, student trained on real+pseudo loss, round-2 iteration with v46 as new teacher → v47",
    });
    s.addText("Fig. 2  Same teacher, two ways to use it. Hard discards 84 % of test cells. Soft keeps everything, weighted by teacher confidence. Round 2 uses v46 as the new teacher → v47.",
      {
        x: 0.30, y: 4.75, w: 9.40, h: 0.35,
        fontFace: "Calibri", fontSize: 8.5, italic: true, color: MUTED, margin: 0,
      });
  }

  footer(s, 7);
  s.addNotes(
    "Act 5 was the breakthrough. Three experiments, +0.067 LB combined.\n\n" +
    "v44 added Lee 2013 hard pseudo-labels. Teacher v41 predicts on the test set. Keep cells " +
    "where p < 0.05 or p > 0.95 — about 9,350 of 59,040 cells (16 %). Hard 0/1 labels, " +
    "patient_id = -1 sentinel so MIL skips them. +0.025 LB.\n\n" +
    "v46 was the bigger insight. Hard discards 84 % of test cells. Switch to Hinton 2015 soft-target " +
    "distillation: use ALL 59,040 cells, target = teacher's raw probability. Preserves what Hinton " +
    "calls 'dark knowledge' — the directional and magnitude information of teacher uncertainty " +
    "that hard binarization throws away. Pseudo loss weight 0.5 so real labels still dominate the " +
    "gradient. +0.039 LB more.\n\n" +
    "v47 was the diminishing-returns check. Xie 2020 iterative noisy student: same recipe as v46 " +
    "but v46's ensemble (LB 0.8236) is the new teacher. Round 2 of the teacher → student loop. " +
    "Forecast +0.005-0.020. Observed +0.003. The round-1 step did the bulk of the work; round 2 " +
    "is mostly a variance reducer (per-seed tr_auc tightened from 0.989-0.993 to 0.993-0.994)."
  );
}

// =========================================================================
// === SLIDE 9: §7 — HARD vs SOFT (with Fig. 3 histogram) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 7, "Hard vs Soft Pseudo-labels", 9);
  slideTitle(s, "Hard drops 84% of test cells; soft keeps all");
  slideDek(s, "Same teacher. Same predictions. Different way of using them.");

  // Comparison table (top-left)
  s.addText("METHOD", {
    x: 0.30, y: 1.65, w: 1.80, h: 0.25,
    fontFace: "Calibri", fontSize: 9, bold: true, color: TEAL, charSpacing: 3, margin: 0,
  });
  s.addText("CELLS USED", {
    x: 2.10, y: 1.65, w: 1.80, h: 0.25,
    fontFace: "Calibri", fontSize: 9, bold: true, color: TEAL, charSpacing: 3, margin: 0,
  });
  s.addText("Δ LB", {
    x: 3.90, y: 1.65, w: 1.00, h: 0.25,
    fontFace: "Calibri", fontSize: 9, bold: true, color: TEAL, charSpacing: 3, margin: 0,
  });
  // Three rows
  const rows = [
    { method: "baseline v41",  cells: "114,302", delta: "—",      color: MUTED },
    { method: "v44 · hard",    cells: "+ 9,350  (16 %)",  delta: "+0.025", color: EMERALD },
    { method: "v46 · soft",    cells: "+ 59,040 (100 %)", delta: "+0.039", color: AMBER },
    { method: "v47 · round 2", cells: "+ 59,040 (100 %)", delta: "+0.003", color: TEAL },
  ];
  let ry = 1.90;
  rows.forEach((r, idx) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.30, y: ry, w: 4.60, h: 0.04,
      fill: { color: LIGHT }, line: { color: LIGHT, width: 0 },
    });
    s.addText(r.method, {
      x: 0.30, y: ry + 0.05, w: 1.80, h: 0.30,
      fontFace: "Calibri", fontSize: 10.5, bold: true, color: NAVY, margin: 0,
    });
    s.addText(r.cells, {
      x: 2.10, y: ry + 0.05, w: 1.80, h: 0.30,
      fontFace: "Calibri", fontSize: 10.5, color: MUTED, margin: 0,
    });
    s.addText(r.delta, {
      x: 3.90, y: ry + 0.05, w: 1.00, h: 0.30,
      fontFace: "Georgia", fontSize: 12, bold: true, color: r.color, margin: 0,
    });
    ry += 0.45;
  });

  // Dark-knowledge explanation card (mid-left)
  calloutQuote(s, 0.30, 3.85, 4.60, 1.10,
    "Dark knowledge (Hinton 2015)",
    "A cell labelled p = 0.78 by the teacher carries directional + magnitude information that hard binarisation discards. Soft targets preserve all of it.",
    { quoteSize: 10, accent: AMBER });

  // Fig. 3 histogram on the right
  if (fs.existsSync(FIG_HIST)) {
    s.addImage({
      path: FIG_HIST,
      x: 5.00, y: 1.55, w: 4.85, h: 1.95,
      altText: "Distribution of v46 teacher predictions over 59,040 test cells. Red dashed lines at p=0.05 and p=0.95 mark the hard-pseudo thresholds. The middle 84.1% of cells (49,636) is the soft-only 'dark knowledge' zone.",
    });
    s.addText("Fig. 3  Distribution of teacher predictions. Red dashed = hard thresholds. Soft keeps every column.",
      {
        x: 5.00, y: 3.55, w: 4.85, h: 0.30,
        fontFace: "Calibri", fontSize: 8.5, italic: true, color: MUTED, margin: 0,
      });
  }

  // Bottom takeaway
  calloutQuote(s, 5.00, 3.95, 4.70, 1.00,
    "Why soft also helps round 2",
    "v47's +0.003 lift on top of v46's +0.039 is small in mean — but per-seed tr_auc tightened (0.989-0.993 → 0.993-0.994). Round-2 acts as a variance reducer.",
    { quoteSize: 9.5, accent: TEAL });

  footer(s, 7);
  s.addNotes(
    "Hard vs soft pseudo-labels, drilled down. Same teacher. Same 59,040 predictions on test cells. " +
    "Two different ways of using them.\n\n" +
    "The histogram on the right is the actual distribution of v46's predictions across all test " +
    "cells. The red dashed lines are the hard-pseudo thresholds at p = 0.05 and p = 0.95. The grey " +
    "tail-shaded regions are where hard pseudo keeps cells — only 9,404 of 59,040, about 16 %.\n\n" +
    "Hinton's 'dark knowledge' lives in the middle 84 %. A cell labelled p = 0.78 by the teacher " +
    "tells the student more than a hard 1.0 does — it carries directional and magnitude information " +
    "about teacher uncertainty. Hard binarization throws all that away.\n\n" +
    "v47's round-2 doesn't add much mean (+0.003) but it tightens the per-seed tr_auc band " +
    "(0.989-0.993 in v46 → 0.993-0.994 in v47). So it acts as a variance reducer — useful for the " +
    "private-LB shake-out."
  );
}

// =========================================================================
// === SLIDE 10: §8 — EVIDENCE (LB chart) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 8, "Evidence", 10);
  slideTitle(s, "The whole story in one chart");

  if (fs.existsSync(FIG_LB)) {
    s.addImage({
      path: FIG_LB,
      x: 0.30, y: 1.50, w: 9.40, h: 3.40,
      altText: "Public LB progression across 30+ logged Kaggle submissions: v19 baseline 0.7455 → v47_s2 distillation seed 0.8355 → v58 intermediate co-attention 0.8392, the new best (peaked #1, currently #3). v59 (EfficientNetV2-S) and v60 (192px) regressed; the v61 2-seed average sits below the best seed. Dashed red line marks the current leader at 0.8448.",
    });
    s.addText("Fig. 4  Public LB across 30+ logged submissions. Amber = co-attention (new best) · green = pseudo-label breakthrough · blue = useful gain · red = regression · grey = neutral. Dashed red = current leader.",
      {
        x: 0.30, y: 4.95, w: 9.40, h: 0.35,
        fontFace: "Calibri", fontSize: 9, italic: true, color: MUTED, margin: 0,
      });
  }

  footer(s, 8);
  s.addNotes(
    "This is the whole story in one image. X-axis: chronological version order. Y-axis: public LB AUC. " +
    "Color-coding: green for breakthroughs, blue for useful gains, red for regressions, grey for neutral.\n\n" +
    "Read left to right:\n" +
    "- v19 at 0.7455 sets the supervised floor (dotted blue line).\n" +
    "- v42 at 0.59 is the SSL collapse — took weeks to diagnose.\n" +
    "- v43 at 0.7444 is the stacked-changes regression.\n" +
    "- v44 at 0.7812 is the hard-pseudo breakthrough.\n" +
    "- v45_probe at 0.7729 confirms cross-recipe ensembles fail at 0.025 LB gaps.\n" +
    "- v46 at 0.8236 is the distillation breakthrough.\n" +
    "- v47 (iterative noisy student round 2, Xie 2020): 0.8264 ensemble, 0.8355 best single seed. We peaked #1 here; two teams later overtook us, so we now sit #3.\n" +
    "- The right-edge cluster is the final single-model architecture sweep: v58 intermediate co-attention reached 0.8392 (the new best, amber); v59 (EfficientNetV2-S) and v60 (192px) both regressed; and v61's 2-seed co-attention average (0.8308) landed below the best seed -- ensembling failing yet again.\n\n" +
    "The +0.094 LB lift from v19 (0.7455) to v58's co-attention best (0.8392) is the headline -- " +
    "the next two slides cover why ensembling never helped, and the co-attention final push that set that best."
  );
}

// =========================================================================
// === SLIDE 11: §9 — NEGATIVE RESULTS GRID ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 9, "Negative Results", 11);
  slideTitle(s, "Six diagnosed failures that shaped the winning recipe");

  // 2x3 grid of failure cards with RULE → entries
  const failures = [
    {
      x: 0.30, y: 1.55, ver: "v22",
      head: "Cross-recipe ensemble collapse",
      diag: "Sigmoid-avg of v19 + v21 at a 0.05 LB gap regressed below v19 alone (0.7422 vs 0.7455).",
      rule: "Don't ensemble across recipes beyond a 0.02 LB gap.",
    },
    {
      x: 3.45, y: 1.55, ver: "v27",
      head: "Held-out val too noisy on N=12",
      diag: "2-patient holdout (pat_5 saturated FL, pat_14 dark FL) selected best_epoch = 0 — untrained.",
      rule: "Use public LB as external validator; abandon held-out splits.",
    },
    {
      x: 6.60, y: 1.55, ver: "v37 / v38",
      head: "Lian 2024 transfer attempts",
      diag: "Heavy FL aug + early fusion both regressed (−0.036, −0.031). Lian's 4-ch ≠ our 1-ch grayscale.",
      rule: "Re-validate any borrowed ablation on the actual modality.",
    },
    {
      x: 0.30, y: 3.30, ver: "v42",
      head: "SSL patient-shortcut collapse",
      diag: "CoMIR InfoNCE solved by patient-ID (loss = 0.02 in epoch 1). tr_auc 0.96 · LB 0.59.",
      rule: "Cross-modal SSL amplifies patient-grouped shortcuts.",
    },
    {
      x: 3.45, y: 3.30, ver: "v43",
      head: "Stacked regulariser regression",
      diag: "Four simultaneous changes on top of v41 → −0.012 LB. No attribution possible.",
      rule: "One or two related variables per submission.",
    },
    {
      x: 6.60, y: 3.30, ver: "v45_probe",
      head: "Ensemble rule confirmation",
      diag: "Sigmoid-avg of v41 + v44 (0.025 LB gap) regressed 0.008 below v44. Pearson 0.936.",
      rule: "Tightened the rule from §22 to ≤ 0.015 LB gap.",
    },
  ];

  failures.forEach(({ x, y, ver, head, diag, rule }) => {
    // Card frame
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 3.10, h: 1.65,
      fill: { color: WHITE }, line: { color: LIGHT, width: 0.6 },
      shadow: { type: "outer", color: "000000", opacity: 0.05, blur: 5, offset: 1.5, angle: 90 },
    });
    // Top crimson stripe
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 3.10, h: 0.06,
      fill: { color: CRIMSON }, line: { color: CRIMSON, width: 0 },
    });
    // Version tag
    s.addText(ver, {
      x: x + 0.15, y: y + 0.12, w: 1.80, h: 0.28,
      fontFace: "Georgia", fontSize: 12.5, bold: true, color: CRIMSON, margin: 0,
    });
    // Headline
    s.addText(head, {
      x: x + 0.15, y: y + 0.38, w: 2.80, h: 0.28,
      fontFace: "Calibri", fontSize: 10.5, bold: true, color: NAVY, margin: 0,
    });
    // Diagnosis
    s.addText(diag, {
      x: x + 0.15, y: y + 0.64, w: 2.80, h: 0.60,
      fontFace: "Calibri", fontSize: 9, color: SLATE, margin: 0,
    });
    // RULE arrow + text
    s.addText("RULE →", {
      x: x + 0.15, y: y + 1.27, w: 0.85, h: 0.30,
      fontFace: "Calibri", fontSize: 8.5, bold: true, color: AMBER,
      charSpacing: 2, margin: 0,
    });
    s.addText(rule, {
      x: x + 1.00, y: y + 1.27, w: 2.00, h: 0.30,
      fontFace: "Calibri", fontSize: 8.5, italic: true, color: SLATE,
      margin: 0,
    });
  });

  // (Synthesis line removed — the 6 cards + title carry it; bottom band was too tight.
  //  The synthesis message lives in the speaker notes.)

  footer(s, 9);
  s.addNotes(
    "Six diagnosed failures. Each one fed forward into the v46 / v47 winning recipe.\n\n" +
    "v22: cross-recipe sigmoid-avg at 0.05 LB gap regressed below v19 alone. Rule encoded.\n\n" +
    "v27: held-out 2-patient validation on N=12 selected patients that turned out to be at FL " +
    "exposure extremes. Best-epoch policy locked onto random init.\n\n" +
    "v37/v38: tried to transfer Lian 2024's modality-specific FL aug and early fusion. Their setup " +
    "is 4-channel emission stacks at higher resolution; ours is 1-channel grayscale at 128. Didn't transfer.\n\n" +
    "v42: the SSL collapse — patient shortcut. InfoNCE loss was 0.02 by epoch 1, suspicious. " +
    "Diagnosed: contrastive task solved by encoding patient identity.\n\n" +
    "v43: four-stacked-changes regression. Confounded experiment. No attribution.\n\n" +
    "v45_probe: cross-recipe sigmoid-avg at a tighter 0.025 LB gap. Still regressed. Tightened " +
    "the rule to ≤0.015 gap.\n\n" +
    "Every one of these has a one-line rule that propagated forward. The wins are the losses, processed."
  );
}

// =========================================================================
// === SLIDE 12: §10 — WHERE THIS POINTS (forward R&D) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 10, "Ensembling failed", 12);
  slideTitle(s, "Every ensemble I tried lost to the best single model");

  statCard(s, 0.30, 1.70, 3.00, 1.30, "0.8264",
    "3-seed average — BELOW the 0.8355 best single seed", CRIMSON,
    { bigSize: 30, smallSize: 9.5 });
  statCard(s, 3.50, 1.70, 3.00, 1.30, "0.7074",
    "orthogonal feature-GBM blend — CV said 0.82, the LB said this", CRIMSON,
    { bigSize: 30, smallSize: 9.5 });
  statCard(s, 6.70, 1.70, 3.00, 1.30, "0.7422",
    "cross-recipe average (v22) — below v19 (0.7455) alone", CRIMSON,
    { bigSize: 30, smallSize: 9.5 });

  bulletList(s, 0.30, 3.30, 9.40, 1.55, [
    { head: "Within-recipe: averaging regresses — confirmed on two recipes.",
      body: "v47's 3-seed mean (0.8264) fell below its best seed (0.8355); v61's 2-seed co-attention mean (0.8308) fell below its best seed (0.8392). Per-seed LB spans ~0.023 on N=12 — the variance is signal-destroying, not noise to average out." },
    { head: "Cross-model: the stronger member is always pulled down.",
      body: "Symmetric averaging drags toward the weaker member, and on this data the quality gap is always wide (v22, v45_probe, and the GBM blend all regressed)." },
    { head: "Even an orthogonal model couldn't save it.",
      body: "A decorrelated feature-GBM (rank-corr 0.41) looked perfect on paper — but its CV (0.82) did not transfer (LB ~0.60). CV-vs-LB struck the ensemble too." },
  ], { accent: CRIMSON, size: 10.5 });

  calloutStrip(s, 0.30, 4.92, 9.40, "Take-home",
    "On N=12, variance dominates — any averaging regresses below the best single seed. Submit one, never a blend.",
    { h: 0.32, quoteSize: 10, accent: CRIMSON });

  footer(s, 10);
  s.addNotes(
    "This is the slide I most want you to remember: on this problem, ENSEMBLING NEVER WORKED. " +
    "Not once. I tried every flavour and every single one regressed below my best single model.\n\n" +
    "Within-recipe seed averaging: my three v47 seeds average to 0.8264, but the best single seed is " +
    "0.8355. I confirmed it a second time on the winning co-attention recipe — v61's two-seed average " +
    "scored 0.8308, below its best seed 0.8392. The per-seed leaderboard spread is ~0.023 on twelve " +
    "patients — so wide that averaging doesn't cancel noise, it just drags the lucky draw back down to the mean.\n\n" +
    "Cross-recipe averaging: v22 averaged v19 and v21 and landed at 0.7422, below v19's 0.7455 alone. " +
    "v45_probe confirmed it. Symmetric averaging always pulls toward the weaker member, and the quality " +
    "gap between any two of my models is wide enough that it hurts.\n\n" +
    "And the one that should have worked: a gradient-boosted tree on hand-crafted features, genuinely " +
    "decorrelated from the CNN at rank-correlation 0.41 — textbook ideal ensemble ingredient. Its " +
    "leave-patients-out CV was 0.82. Blended 50/50 with my best model it scored 0.7074. Its CV simply " +
    "did not transfer to unseen patients — the exact CV-versus-LB trap that defines this whole project, " +
    "now sprung on the ensemble itself.\n\n" +
    "The conclusion is clean: with twelve patients, variance dominates everything, and any averaging " +
    "regresses below the best single draw. My final submission is one well-chosen single model — never a blend."
  );
}

// =========================================================================
// === SLIDE 13: §11 — THE FINAL PUSH (co-attention + learning curves) ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  sectionHeader(s, 11, "The final push", 13);
  slideTitle(s, "Co-attention broke the plateau — a new best, 0.8392");

  // Learning curves (left) — the teacher explicitly asked for these
  const FIG_LC = path.join(__dirname, "learning_curves_v58.png");
  if (fs.existsSync(FIG_LC)) {
    s.addImage({
      path: FIG_LC,
      x: 0.30, y: 1.62, w: 5.25, h: 3.05,
      sizing: { type: "contain", w: 5.25, h: 3.05 },
      altText: "Training curves for the v58 co-attention model: per-cell train AUC climbs smoothly from ~0.83 to ~0.99 over 12 epochs for both seeds, and loss decreases monotonically with no divergence (fp32 attention + ReZero gate kept it stable).",
    });
    s.addText("Fig. 5  v58 co-attention learning curves — train AUC + loss, 2 seeds, 12 epochs. Clean, no instability.",
      { x: 0.30, y: 4.74, w: 5.25, h: 0.42, fontFace: "Calibri", fontSize: 9, italic: true, color: MUTED, margin: 0 });
  }

  // Architecture sweep (right): what worked vs what didn't
  statCard(s, 5.80, 1.70, 3.90, 0.90, "0.8392  ✓",
    "intermediate CO-ATTENTION (CAFNet-style) — new best, beats v47's 0.8355", TEAL,
    { bigSize: 21, smallSize: 9 });
  statCard(s, 5.80, 2.70, 3.90, 0.90, "0.7823  ✗",
    "higher resolution (192 px) — regressed", CRIMSON,
    { bigSize: 21, smallSize: 9 });
  statCard(s, 5.80, 3.70, 3.90, 0.90, "0.7376  ✗",
    "more capacity (EfficientNetV2-S) — regressed", CRIMSON,
    { bigSize: 21, smallSize: 9 });
  s.addText("Of every architecture tried, only fusion design moved the needle — the model is data-bound. Capacity and resolution both hurt.",
    { x: 5.80, y: 4.74, w: 3.90, h: 0.55, fontFace: "Calibri", fontSize: 9.5, italic: true, color: SLATE, margin: 0 });

  footer(s, 11);
  s.addNotes(
    "The final chapter, after the distillation breakthrough. I swept the remaining architectural " +
    "levers as single EfficientNet models — no ensembles (those all failed; next slide).\n\n" +
    "Two hurt: bumping capacity to EfficientNetV2-S dropped to 0.7376 (a bigger model overfits 12 " +
    "patients), and raising input resolution to 192px dropped to 0.7823. Both reinforce the data-bound thesis.\n\n" +
    "One helped: intermediate co-attention fusion — the CAFNet design from the source-dataset paper, " +
    "applied to EfficientNet-B0. Its best seed reached 0.8392, a new best over v47's 0.8355, and it " +
    "lifted the mean too. The learning curves (left) show clean training to ~0.99 train AUC with no " +
    "instability — I had to harden the attention (fp32 + a zero-init ReZero gate) to stop an fp16 NaN.\n\n" +
    "Take-home: fusion design is the one architectural lever that moved the needle here; capacity and " +
    "resolution did not."
  );
}

// =========================================================================
// === SLIDE 14: §12 — TAKE-HOME ===
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // Teal accent bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.22, h: 5.625,
    fill: { color: TEAL }, line: { color: TEAL, width: 0 },
  });
  // Page indicator
  s.addText("14 / 14", {
    x: 8.30, y: 0.22, w: 1.40, h: 0.30,
    fontFace: "Calibri", fontSize: 10, color: MUTED,
    align: "right", margin: 0,
  });

  s.addText("§12  ·  TAKE-HOME", {
    x: 0.80, y: 0.30, w: 5.0, h: 0.30,
    fontFace: "Calibri", fontSize: 10.5, bold: true,
    color: TEAL, charSpacing: 5, margin: 0,
  });
  s.addText("Five things I'll carry forward", {
    x: 0.80, y: 0.65, w: 8.6, h: 0.65,
    fontFace: "Georgia", fontSize: 28, bold: true, color: WHITE, margin: 0,
  });

  const takeaways = [
    { head: "Higher CV ≠ higher LB.",
      body: "v15 had my best CV (0.866) and my worst LB (0.572). On patient-grouped OOD, CV alone is not a signal." },
    { head: "Don't stack unvalidated changes.",
      body: "v43 added four tweaks at once → −0.012 LB with no attribution. One or two related variables per experiment." },
    { head: "Seed variance is huge.",
      body: "Same recipe, different seed: 0.03 LB shift (v19 vs v23). Single-seed A/B claims under 0.03 are noise." },
    { head: "On small-N, data is the lever — not architecture.",
      body: "Four regularisers gained +0.011; soft-pseudo distillation gained +0.042. Even the SOTA co-attention fusion ≈ plain late fusion (0.97 corr, both 0.83). Only co-attention fusion nudged architecture (+0.004 to a new best 0.8392); data was ~10x bigger." },
    { head: "Negative results are the methodology.",
      body: "The v46/v47 winning recipe traces directly to six diagnosed failures. Reading the failures faithfully is what produced the win." },
  ];

  const y0 = 1.60;
  takeaways.forEach(({ head, body }, i) => {
    const y = y0 + i * 0.66;
    // Number — large amber Georgia
    s.addText(String(i + 1).padStart(2, "0"), {
      x: 0.80, y: y, w: 0.65, h: 0.50,
      fontFace: "Georgia", fontSize: 22, bold: true, color: AMBER,
      align: "left", margin: 0,
    });
    s.addText(head, {
      x: 1.55, y: y, w: 8.10, h: 0.32,
      fontFace: "Calibri", fontSize: 13, bold: true, color: WHITE, margin: 0,
    });
    s.addText(body, {
      x: 1.55, y: y + 0.32, w: 8.10, h: 0.32,
      fontFace: "Calibri", fontSize: 10.5, italic: true, color: CREAM, margin: 0,
    });
  });

  // Footer
  s.addText("Code & LB history  ·  github.com/rafallex/multimodal-microscopy-distillation", {
    x: 0.80, y: 5.10, w: 6.5, h: 0.30,
    fontFace: "Calibri", fontSize: 10.5, italic: true, color: MUTED, margin: 0,
  });
  s.addText("Thank you. Questions?", {
    x: 7.30, y: 5.10, w: 2.4, h: 0.30,
    fontFace: "Calibri", fontSize: 10.5, italic: true, color: AMBER,
    align: "right", margin: 0,
  });

  s.addNotes(
    "Five things to carry forward.\n\n" +
    "One. Higher CV doesn't mean higher LB. v15 had my best cross-validation, 0.866, and my " +
    "worst leaderboard, 0.572. If you only have CV, you don't know what you have. On patient-grouped " +
    "OOD, CV alone is not a signal.\n\n" +
    "Two. Don't stack unvalidated changes. v43 added four tweaks at once on top of v41 and " +
    "regressed by 0.012 LB. The run cost 5 hours of GPU and produced no isolable signal. I now " +
    "change at most one or two related variables per submission.\n\n" +
    "Three. Seed variance is huge. v23 was v19 with only the random seed changed. 0.03 LB shift. " +
    "Most A/B comparisons with deltas under 0.03 are noise, not signal.\n\n" +
    "Four. On 12 patients, dataset size beats regularization. Four textbook regularizers gained " +
    "0.011 LB. Soft pseudo + noisy-student round 2 gained 0.042 LB. About 3.8× larger lift from " +
    "the data lever than the model lever.\n\n" +
    "Five. Negative results are the methodology. The v46 and v47 winning recipes are direct " +
    "responses to six diagnosed failures: v22, v27, v37, v38, v42, v43, and v45_probe. Reading the " +
    "failures faithfully is what produced the win.\n\n" +
    "Thanks. Full code and version-by-version LB history at github.com/rafallex/multimodal-microscopy-distillation. Questions?"
  );
}

// =========================================================================
// === Save ===
// =========================================================================
const OUT = path.join(__dirname, "A3_cancer_challenge.pptx");
pres.writeFile({ fileName: OUT }).then(() => {
  console.log("Saved:", OUT);
}).catch(err => {
  console.error("Build error:", err);
  process.exit(1);
});
