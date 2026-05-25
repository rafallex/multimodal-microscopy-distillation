/**
 * A3_cancer_challenge_claude.pptx — 12-slide deck built via pptxgenjs.
 *
 * Topic: Multimodal Cancer Cell Classification (Uppsala 1MD042 A3).
 * Current public-LB: #1 at 0.8236 with v46 (Hinton-style soft-pseudo distillation).
 *
 * Palette: "Microscopy Indigo" — content-informed for medical imaging.
 *   PRIMARY  #1A3A52  deep slate-navy (dark backgrounds, body text on cream)
 *   TEAL     #0D9488  fluorescence cyan (motif accent, links)
 *   AMBER    #D97706  highlight / breakthrough callouts
 *   EMERALD  #059669  wins / positive outcomes
 *   CRIMSON  #B91C1C  regressions / negative outcomes
 *   CREAM    #FAF7F2  body background
 *   MUTED    #64748B  secondary text, axis labels
 *   WHITE    #FFFFFF
 *
 * Typography: Georgia (headers) + Calibri (body). Title-card teal accent dot motif
 * appears in upper-left of every content slide for visual continuity.
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
const SLATE   = "334155";  // for darker body text on cream

// === Layout ===
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";   // 10" × 5.625"
pres.author = "Rafael Tavares Proença";
pres.title  = "Multimodal Cancer Cell Classification";

const REPO_ROOT = path.resolve(__dirname, "..");
const LB_CHART = path.join(REPO_ROOT, "report", "figures", "lb_progression.png");

// === Reusable elements ===
function dotMotif(slide) {
  // Top-left teal dot — visual continuity across content slides
  slide.addShape(pres.shapes.OVAL, {
    x: 0.30, y: 0.30, w: 0.18, h: 0.18,
    fill: { color: TEAL }, line: { color: TEAL, width: 0 },
  });
}

function eyebrow(slide, text, y = 0.30) {
  slide.addText(text.toUpperCase(), {
    x: 0.62, y: y, w: 9, h: 0.30,
    fontFace: "Calibri", fontSize: 10.5, bold: true,
    color: TEAL, charSpacing: 4, margin: 0,
  });
}

function title(slide, text, y = 0.65) {
  slide.addText(text, {
    x: 0.30, y: y, w: 9.40, h: 0.85,
    fontFace: "Georgia", fontSize: 28, bold: true,
    color: NAVY, margin: 0,
  });
}

function footer(slide, page, total = 12) {
  slide.addText("Multimodal Cancer Cell Classification  ·  Uppsala 1MD042 A3", {
    x: 0.30, y: 5.32, w: 6.5, h: 0.22,
    fontFace: "Calibri", fontSize: 9, italic: true, color: MUTED, margin: 0,
  });
  slide.addText(`${page} / ${total}`, {
    x: 8.50, y: 5.32, w: 1.20, h: 0.22,
    fontFace: "Calibri", fontSize: 9, color: MUTED, align: "right", margin: 0,
  });
}

function statCard(slide, x, y, w, h, big, small, bigColor = NAVY) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: WHITE }, line: { color: "E5E7EB", width: 0.5 },
    shadow: { type: "outer", color: "000000", opacity: 0.06, blur: 6, offset: 2, angle: 90 },
  });
  // Teal accent stripe on left
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.07, h,
    fill: { color: TEAL }, line: { color: TEAL, width: 0 },
  });
  slide.addText(big, {
    x: x + 0.22, y: y + 0.10, w: w - 0.30, h: h * 0.50,
    fontFace: "Georgia", fontSize: 28, bold: true, color: bigColor,
    align: "left", valign: "top", margin: 0,
  });
  if (small) {
    slide.addText(small, {
      x: x + 0.22, y: y + h * 0.55, w: w - 0.30, h: h * 0.40,
      fontFace: "Calibri", fontSize: 10.5, color: MUTED,
      align: "left", valign: "top", margin: 0,
    });
  }
}

// =========================================================================
// Slide 1: TITLE
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // Teal accent bar on left edge
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.22, h: 5.625,
    fill: { color: TEAL }, line: { color: TEAL, width: 0 },
  });

  s.addText("ASSIGNMENT 3  ·  KAGGLE CHALLENGE", {
    x: 0.80, y: 0.85, w: 8.5, h: 0.30,
    fontFace: "Calibri", fontSize: 11.5, bold: true, color: TEAL,
    charSpacing: 5, margin: 0,
  });

  s.addText("Multimodal Cancer Cell\nClassification", {
    x: 0.80, y: 1.30, w: 8.6, h: 1.70,
    fontFace: "Georgia", fontSize: 42, bold: true, color: WHITE,
    paraSpaceBefore: 0, margin: 0,
  });

  s.addText("From SSL collapse to #1 on the public leaderboard", {
    x: 0.80, y: 3.15, w: 8.6, h: 0.45,
    fontFace: "Georgia", fontSize: 17, italic: true, color: CREAM,
    margin: 0,
  });

  // Headline stat — large amber callout
  s.addText("+0.078 LB across 29 versions", {
    x: 0.80, y: 3.80, w: 6.5, h: 0.42,
    fontFace: "Georgia", fontSize: 16, bold: true, color: AMBER,
    margin: 0,
  });

  // Author / course block
  s.addText("Rafael Tavares Proença", {
    x: 0.80, y: 4.35, w: 8.6, h: 0.32,
    fontFace: "Calibri", fontSize: 14, bold: true, color: WHITE, margin: 0,
  });
  s.addText("Advanced Deep Learning for Image Processing (1MD042)", {
    x: 0.80, y: 4.65, w: 8.6, h: 0.28,
    fontFace: "Calibri", fontSize: 12, color: CREAM, margin: 0,
  });
  s.addText("Uppsala University  ·  June 4, 2026", {
    x: 0.80, y: 4.93, w: 8.6, h: 0.28,
    fontFace: "Calibri", fontSize: 11, color: MUTED, margin: 0,
  });

  s.addNotes(
    "Hello — I'm Rafael. Today I'll walk you through the Multimodal Cancer " +
    "Cell Classification Challenge: 29 logged Kaggle versions, five acts of " +
    "experimentation, and a +0.078 LB lift over the supervised baseline. " +
    "We currently hold #1 on the public leaderboard at 0.8236. " +
    "But this talk isn't about the score — the instructor explicitly told us " +
    "grades are based on methodology and presentation, not on the leaderboard. " +
    "So I'll spend most of the time on what didn't work and why, since the " +
    "diagnosed failures were what shaped the version that won."
  );
}

// =========================================================================
// Slide 2: THE CHALLENGE
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  dotMotif(s);
  eyebrow(s, "The challenge");
  title(s, "Binary classification on paired microscopy");

  // Left column: setup bullets
  s.addText([
    { text: "Per-cell label: malignant vs benign (oral cancer)", options: { bullet: true, breakLine: true } },
    { text: "Inputs: paired BF + FL microscopy, 128×128 grayscale", options: { bullet: true, breakLine: true } },
    { text: "12 training patients, patient-disjoint test set", options: { bullet: true, breakLine: true } },
    { text: "OOD generalization is the dominant failure mode", options: { bullet: true, breakLine: true } },
    { text: "Grading: methodology over leaderboard score", options: { bullet: true, color: AMBER, bold: true } },
  ], {
    x: 0.40, y: 1.55, w: 5.20, h: 3.30,
    fontFace: "Calibri", fontSize: 14, color: SLATE,
    paraSpaceAfter: 8, margin: 0,
  });

  // Right column: stat cards
  statCard(s, 5.95, 1.55, 3.70, 1.10, "12", "training patients");
  statCard(s, 5.95, 2.75, 3.70, 1.10, "2 modalities", "brightfield + fluorescence");
  statCard(s, 5.95, 3.95, 1.78, 0.95, "128×", "input side");
  statCard(s, 7.87, 3.95, 1.78, 0.95, "#1", "current public LB", AMBER);

  footer(s, 2);
  s.addNotes(
    "The setup. Per-cell binary classification — malignant vs benign oral cancer cells — " +
    "from paired brightfield and fluorescence microscopy, both 128×128 grayscale.\n\n" +
    "Two structural challenges. First: only 12 training patients. Small N. " +
    "Second: test patients are completely unseen — strict patient-disjoint OOD. " +
    "That second constraint dominates everything that follows.\n\n" +
    "Anchor for the audience: we currently hold #1 on the public LB at 0.8236. The amber " +
    "card on the right is our current position. The path to get there is the rest of this talk."
  );
}

// =========================================================================
// Slide 3: 5 ACTS TIMELINE
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  dotMotif(s);
  eyebrow(s, "The journey");
  title(s, "29 versions, 5 acts, one breakthrough");

  // Horizontal timeline
  const TY = 2.90;     // Y of timeline
  const TX = 0.70;     // X start
  const TW = 8.60;     // total width
  s.addShape(pres.shapes.RECTANGLE, {
    x: TX, y: TY, w: TW, h: 0.05,
    fill: { color: MUTED, transparency: 50 }, line: { color: MUTED, width: 0 },
  });

  // 5 act markers
  const acts = [
    { px: 1.20, name: "Act 1+2", vers: "v10–v16", label: "Baselines & SSL",    color: CRIMSON, result: "0.572",  note: "SSL collapse" },
    { px: 3.20, name: "Act 3",    vers: "v17–v19", label: "EffNet pivot",       color: NAVY,    result: "0.7455", note: "supervised floor" },
    { px: 5.20, name: "Act 4",    vers: "v20–v41", label: "Regularizer search", color: NAVY,    result: "0.7563", note: "v41 +0.011" },
    { px: 7.20, name: "Act 5",    vers: "v42–v44", label: "Pseudo-labels",      color: EMERALD, result: "0.7812", note: "v44 +0.025" },
    { px: 9.10, name: "Climax",   vers: "v46",     label: "Distillation",       color: AMBER,   result: "0.8236", note: "v46 #1 on LB" },
  ];

  acts.forEach(({ px, name, vers, label, color, result, note }) => {
    // Marker dot
    s.addShape(pres.shapes.OVAL, {
      x: px - 0.14, y: TY - 0.13, w: 0.32, h: 0.32,
      fill: { color }, line: { color: WHITE, width: 1.5 },
    });
    // Act name (above dot)
    s.addText(name.toUpperCase(), {
      x: px - 0.80, y: 1.65, w: 1.60, h: 0.25,
      fontFace: "Calibri", fontSize: 9.5, bold: true, color: TEAL,
      align: "center", margin: 0, charSpacing: 2,
    });
    // Version range
    s.addText(vers, {
      x: px - 0.80, y: 1.95, w: 1.60, h: 0.30,
      fontFace: "Georgia", fontSize: 13, bold: true, color: NAVY,
      align: "center", margin: 0,
    });
    // Theme label
    s.addText(label, {
      x: px - 0.90, y: 2.30, w: 1.80, h: 0.35,
      fontFace: "Calibri", fontSize: 10.5, italic: true, color: MUTED,
      align: "center", margin: 0,
    });
    // Result number (below dot)
    s.addText(result, {
      x: px - 0.90, y: 3.40, w: 1.80, h: 0.35,
      fontFace: "Georgia", fontSize: 15, bold: true, color,
      align: "center", margin: 0,
    });
    // Result note
    s.addText(note, {
      x: px - 0.95, y: 3.78, w: 1.90, h: 0.30,
      fontFace: "Calibri", fontSize: 9.5, italic: true, color: MUTED,
      align: "center", margin: 0,
    });
  });

  // Bottom banner
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.30, y: 4.45, w: 9.40, h: 0.60,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText(
    "Each act answered a question. Each result changed the next question.", {
      x: 0.50, y: 4.50, w: 9.20, h: 0.50,
      fontFace: "Georgia", fontSize: 13, italic: true, color: CREAM,
      align: "left", valign: "middle", margin: 0,
    });

  footer(s, 3);
  s.addNotes(
    "Five acts, anchored to version numbers. We tracked every Kaggle submission in a " +
    "LB_HISTORY.md file in the repo, so when I say 'v19' or 'v46' I mean a specific " +
    "notebook with a specific recipe.\n\n" +
    "Act 1+2, v10 through v16, was early iteration and an ambitious CoMIR-style SSL attempt. " +
    "v15 scored 0.866 in cross-validation — our highest CV ever — then collapsed to 0.572 on " +
    "the leaderboard. That single experiment taught us patient-grouped OOD breaks naive CV.\n\n" +
    "Act 3, v17–v19, was the pivot back to a simple EffNet-B0 supervised recipe. LB 0.7455.\n\n" +
    "Act 4, v20–v41, was the regularizer search. v41 stacked four L4-textbook regularizers cleanly " +
    "for +0.011 to 0.7563.\n\n" +
    "Act 5 was the breakthrough — pseudo-labels and distillation. v44 added Lee 2013 pseudo-labels " +
    "for +0.025. Then v46 swapped to Hinton 2015 soft-target distillation for another +0.039. " +
    "We landed at 0.8236, taking #1 on the public LB."
  );
}

// =========================================================================
// Slide 4: ACT 2 — THE SSL COLLAPSE
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  dotMotif(s);
  eyebrow(s, "Act 1+2  ·  the ambition");
  title(s, "v15 / v16: SSL had the best CV — and the worst LB");

  // Left: what we built
  s.addText("What we built", {
    x: 0.40, y: 1.55, w: 4.80, h: 0.30,
    fontFace: "Calibri", fontSize: 11.5, bold: true, color: TEAL, charSpacing: 2, margin: 0,
  });
  s.addText([
    { text: "Two ResNet-18 branches (BF + FL) with projection heads", options: { bullet: true, breakLine: true } },
    { text: "CoMIR-style cross-modal contrastive SSL (NT-Xent, τ=0.1)", options: { bullet: true, breakLine: true } },
    { text: "Supervised stage with discriminative LR (1:10 backbone:head)", options: { bullet: true, breakLine: true } },
    { text: "3-fold stratified-group cross-validation", options: { bullet: true, color: CRIMSON } },
  ], {
    x: 0.40, y: 1.90, w: 4.80, h: 2.40,
    fontFace: "Calibri", fontSize: 13, color: SLATE,
    paraSpaceAfter: 8, margin: 0,
  });

  // Right: the shock — two stacked stat cards
  s.addText("Then the leaderboard arrived", {
    x: 5.50, y: 1.55, w: 4.20, h: 0.30,
    fontFace: "Calibri", fontSize: 11.5, bold: true, color: TEAL, charSpacing: 2, margin: 0,
  });

  // CV card
  statCard(s, 5.50, 1.95, 4.20, 1.10, "0.866 CV AUC", "best we'd ever seen", EMERALD);
  // LB card
  statCard(s, 5.50, 3.15, 4.20, 1.10, "0.572 LB", "near random — patient shortcut", CRIMSON);

  // Diagnosis caption
  s.addText("Diagnosis: contrastive task could be solved by encoding patient identity (≥95% accuracy) — backbone learned the shortcut.", {
    x: 0.40, y: 4.50, w: 9.30, h: 0.50,
    fontFace: "Calibri", fontSize: 11.5, italic: true, color: AMBER,
    margin: 0,
  });

  footer(s, 4);
  s.addNotes(
    "Act 2 is the canonical 'too clever' moment. CoMIR-style cross-modal contrastive SSL: " +
    "two encoders for BF and FL, paired images of the same cell as positives, unpaired as negatives. " +
    "Pretrain 10 epochs, finetune 8.\n\n" +
    "The CV looked spectacular. 0.866 AUC. We'd never seen that high. We submitted to Kaggle, " +
    "expecting 0.75 or so. We got 0.572. Essentially random.\n\n" +
    "Diagnosis. Under our data structure, every cell has exactly one source patient. " +
    "Paired BF+FL of cell i ALWAYS share a patient. Unpaired BF+FL of cell i and j with i≠j " +
    "USUALLY don't share a patient. So the contrastive task can be solved at ≥95% accuracy by " +
    "encoding patient identity rather than cell content. The pretrain backbone optimized that " +
    "shortcut. The supervised stage memorized within-patient class distributions. The test patients " +
    "were different patients. Collapse.\n\n" +
    "Lesson: SSL on a strongly patient-grouped dataset amplifies the patient shortcut. " +
    "To use it as a generalization aid, positive pairs would need to span patient boundaries — " +
    "which our annotations don't support."
  );
}

// =========================================================================
// Slide 5: ACT 3 — THE EFFNET PIVOT
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  dotMotif(s);
  eyebrow(s, "Act 3  ·  the pivot");
  title(s, "Strip back to EffNet-B0 fundamentals");

  // Single big stat card on the right
  statCard(s, 6.20, 1.60, 3.45, 1.80, "0.7455", "v19 public LB — our new floor", EMERALD);

  // Five ingredients — narrow bullets on left
  s.addText("Five ingredients in v19:", {
    x: 0.40, y: 1.60, w: 5.50, h: 0.32,
    fontFace: "Calibri", fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0,
  });
  s.addText([
    { text: "EfficientNet-B0 dual-branch (BF + FL), late concat fusion", options: { bullet: true, breakLine: true } },
    { text: "Per-patient MIL auxiliary loss (mean-logit BCE, weight 0.5)", options: { bullet: true, breakLine: true } },
    { text: "AdaBN test-time BN refresh + test-set stain normalization", options: { bullet: true, breakLine: true } },
    { text: "Strong augmentation + RandomErasing (p=0.25)", options: { bullet: true, breakLine: true } },
    { text: "8-way D4 group test-time augmentation", options: { bullet: true } },
  ], {
    x: 0.40, y: 2.00, w: 5.50, h: 2.50,
    fontFace: "Calibri", fontSize: 12, color: SLATE,
    paraSpaceAfter: 8, margin: 0,
  });

  // Bottom-right: the win
  statCard(s, 6.20, 3.55, 3.45, 0.90, "+0.29 LB", "vs v15 SSL collapse", AMBER);

  // Key-insight bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.30, y: 4.65, w: 9.40, h: 0.42,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Lesson: simpler + OOD-aware (AdaBN, stain norm) beats elaborate SSL on patient-grouped data.", {
    x: 0.50, y: 4.68, w: 9.20, h: 0.38,
    fontFace: "Calibri", fontSize: 11.5, italic: true, color: CREAM,
    align: "left", valign: "middle", margin: 0,
  });

  footer(s, 5);
  s.addNotes(
    "Act 3 was the pivot. Drop the SSL complexity. Drop the leave-one-patient-out CV. " +
    "Train a standard EfficientNet-B0 with five ingredients that directly model the OOD problem.\n\n" +
    "Five ingredients: dual-branch backbone for BF and FL with late concat fusion. " +
    "A per-patient MIL auxiliary loss that aggregates cell logits to the patient level — " +
    "regularizes against pure within-patient memorization. AdaBN to refresh batch-norm statistics " +
    "to the test distribution. Test-set stain normalization, since FL test is measurably 19.7% " +
    "brighter than FL train. Strong augmentation including RandomErasing. And 8-way dihedral TTA.\n\n" +
    "Public LB 0.7455. Up 0.29 from the SSL collapse. v19 became our supervised floor and our " +
    "reference recipe for the rest of the project."
  );
}

// =========================================================================
// Slide 6: ACT 4 — REGULARIZER SEARCH
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  dotMotif(s);
  eyebrow(s, "Act 4  ·  the regularizer search");
  title(s, "Stack one change at a time — v41 wins +0.011");

  // What worked in v41
  s.addText("v41 — four clean additions on top of v19", {
    x: 0.40, y: 1.55, w: 9.20, h: 0.30,
    fontFace: "Calibri", fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0,
  });

  // 4-card row
  const v41Cards = [
    { x: 0.40, big: "ε=0.05", small: "label smoothing" },
    { x: 2.78, big: "0.4",    small: "dropout (from 0.3)" },
    { x: 5.16, big: "paired",  small: "RandomResizedCrop" },
    { x: 7.54, big: "24-way", small: "multiscale D4 TTA" },
  ];
  v41Cards.forEach(({ x, big, small }) => statCard(s, x, 1.95, 2.18, 1.10, big, small));

  // v43 cautionary tale
  s.addText("v43 — four MORE changes stacked at once", {
    x: 0.40, y: 3.30, w: 9.20, h: 0.30,
    fontFace: "Calibri", fontSize: 12, bold: true, color: CRIMSON, charSpacing: 2, margin: 0,
  });

  const v43Cards = [
    { x: 0.40, big: "FL aug", small: "modality-specific" },
    { x: 2.78, big: "WD 3e-4", small: "(was 1e-4)" },
    { x: 5.16, big: "3-seed", small: "× SWA(last 4 ep)" },
    { x: 7.54, big: "40-way", small: "TTA (5 scales)" },
  ];
  v43Cards.forEach(({ x, big, small }) => statCard(s, x, 3.65, 2.18, 0.95, big, small, CRIMSON));

  // Outcome banner
  s.addText("Result: −0.012 LB regression. Confounded. Couldn't isolate the bad component.", {
    x: 0.40, y: 4.75, w: 9.30, h: 0.35,
    fontFace: "Calibri", fontSize: 11.5, italic: true, color: CRIMSON,
    margin: 0,
  });

  footer(s, 6);
  s.addNotes(
    "Act 4 had two contrasting experiments — one clean, one cautionary.\n\n" +
    "v41 was the clean win. Four well-motivated L4-textbook additions on top of v19: " +
    "label smoothing at 0.05, dropout from 0.3 up to 0.4, paired RandomResizedCrop, and " +
    "multiscale TTA — 3 input scales × 8 D4 group elements = 24-way. " +
    "Net: +0.011 LB to 0.7563. Each component additively reasonable, all consistent with the L4 slides.\n\n" +
    "v43 was the cautionary tale. We added four MORE changes simultaneously — FL-tuned augmentation, " +
    "weight decay bump, 3-seed × SWA ensembling, 40-way TTA. " +
    "Result: regressed by 0.012 LB. We had no way to attribute blame. Five hours of Kaggle GPU " +
    "produced no isolable signal. After v43 we adopted a one-or-two-variables-per-experiment rule.\n\n" +
    "Lesson: stacking unvalidated changes is methodological malpractice."
  );
}

// =========================================================================
// Slide 7: ACT 5 — THE SEMI-SUPERVISED PIVOT (text-heavy)
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  dotMotif(s);
  eyebrow(s, "Act 5  ·  the breakthrough");
  title(s, "Pseudo-labels → soft distillation = +0.064 LB");

  // Two big result cards side by side
  statCard(s, 0.40, 1.55, 4.55, 1.55, "v44: +0.025", "Hard pseudo (Lee 2013) — 9k test cells at thr 0.05 / 0.95", EMERALD);
  statCard(s, 5.05, 1.55, 4.60, 1.55, "v46: +0.039", "SOFT pseudo (Hinton 2015) — all 59k cells, raw probs", AMBER);

  // Why it worked
  s.addText("Why distillation amplified the pseudo-label gain", {
    x: 0.40, y: 3.25, w: 9.30, h: 0.30,
    fontFace: "Calibri", fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0,
  });
  s.addText([
    { text: "v44's threshold discarded 84% of test cells. v46 uses all 59,040.", options: { bullet: true, breakLine: true } },
    { text: "Hard pseudo throws away the teacher's confidence; soft preserves it.", options: { bullet: true, breakLine: true } },
    { text: "Effective training set: 114k → 124k (v44) → 173k cells (v46).", options: { bullet: true, breakLine: true } },
    { text: "Pseudo-loss weight 0.5 — real labels still dominate the gradient.", options: { bullet: true } },
  ], {
    x: 0.40, y: 3.55, w: 9.30, h: 1.50,
    fontFace: "Calibri", fontSize: 12, color: SLATE,
    paraSpaceAfter: 6, margin: 0,
  });

  footer(s, 7);
  s.addNotes(
    "Act 5 was the breakthrough. Two experiments, +0.064 LB combined.\n\n" +
    "v44 added Lee 2013-style hard pseudo-labels. Teacher (v41) predicts on the test set; " +
    "keep cells where teacher is >0.95 or <0.05 confident; assign hard 0/1 labels; merge into training. " +
    "Sentinel patient_id = -1 so the MIL loss skips these. About 9,350 cells. +0.025 LB.\n\n" +
    "v46 was the bigger insight. Hard thresholding discards 84% of test cells — only 9k of 59k make the cutoff. " +
    "Switch to Hinton 2015 soft-target distillation. Use ALL 59,000 test cells. " +
    "Use the teacher's raw probability as the BCE target — preserves what Hinton calls 'dark knowledge'. " +
    "Down-weight pseudo loss by 0.5 so real labels still dominate. +0.039 LB more.\n\n" +
    "The combined +0.064 LB lift is six times larger than the +0.011 we got from four textbook " +
    "regularizers. On 12 patients, the dataset-size lever is bigger than the model-quality lever."
  );
}

// =========================================================================
// Slide 8: THE LB CHART (visual hero slide)
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  dotMotif(s);
  eyebrow(s, "The whole story in one chart");
  title(s, "Public LB across 29 logged versions");

  if (fs.existsSync(LB_CHART)) {
    // The chart is roughly 13:6.5 ratio. Make it span most of the slide.
    s.addImage({
      path: LB_CHART,
      x: 0.30, y: 1.55,
      w: 9.40, h: 3.55,
      altText: "LB progression chart showing 29 datapoints from v19 baseline at 0.7455 to v46 #1 at 0.8236",
    });
  } else {
    s.addText("[LB chart not found at " + LB_CHART + "]", {
      x: 0.30, y: 2.50, w: 9.40, h: 1.0,
      fontFace: "Calibri", fontSize: 14, italic: true, color: CRIMSON,
      align: "center", margin: 0,
    });
  }

  footer(s, 8);
  s.addNotes(
    "This is the whole story in one image. X-axis: chronological version order. " +
    "Y-axis: public LB AUC. Color-coding: green for breakthroughs, blue for useful gains, " +
    "red for regressions and negative results.\n\n" +
    "Read it left to right:\n" +
    "- v19 at 0.7455 sets the supervised floor (dotted blue line).\n" +
    "- v42 at 0.59 is the SSL collapse — that one took weeks to diagnose.\n" +
    "- v43 at 0.7444 is the stacked-changes regression.\n" +
    "- v44 at 0.7812 is the hard-pseudo breakthrough.\n" +
    "- v45_probe at 0.7729 confirms cross-recipe ensembles still fail at 0.025 LB gaps.\n" +
    "- v46 at 0.8236 is the distillation breakthrough — current #1 (annotated in green).\n" +
    "- v47 (orange '?') is round 2 of noisy student, training as I present this.\n\n" +
    "The +0.078 lift from v19 to v46 is the headline."
  );
}

// =========================================================================
// Slide 9: WHAT WORKED (positive findings)
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  dotMotif(s);
  eyebrow(s, "What worked");
  title(s, "Three findings that compounded");

  const findings = [
    {
      y: 1.55, color: EMERALD, num: "01",
      head: "Dataset size > regularization on small-N",
      body: "v41 stacked four L4 regularizers for +0.011 LB. v46 added soft pseudo-labels for ~50k extra effective cells: +0.039 LB. The dataset-size lever was 3.5× larger than the model-quality lever.",
    },
    {
      y: 2.65, color: TEAL, num: "02",
      head: "Within-recipe ensembling helps — cross-recipe doesn't",
      body: "3 seeds × SWA averaging inside one recipe gained ~0.005–0.010 LB in v46. Cross-recipe sigmoid-averaging (v22, v45_probe) regressed below the better member at any gap >0.02 LB.",
    },
    {
      y: 3.75, color: NAVY, num: "03",
      head: "Patient identity is adversarial — treat it as such",
      body: "MIL aux loss (patient-grouped), AdaBN, test-stain norm, and a sentinel patient_id = −1 for pseudo cells all explicitly target the patient-shortcut failure mode that broke v42.",
    },
  ];

  findings.forEach(({ y, color, num, head, body }) => {
    // Number badge
    s.addShape(pres.shapes.OVAL, {
      x: 0.40, y, w: 0.85, h: 0.85,
      fill: { color }, line: { color, width: 0 },
    });
    s.addText(num, {
      x: 0.40, y, w: 0.85, h: 0.85,
      fontFace: "Georgia", fontSize: 18, bold: true, color: WHITE,
      align: "center", valign: "middle", margin: 0,
    });

    // Headline + body to the right
    s.addText(head, {
      x: 1.45, y: y - 0.05, w: 8.20, h: 0.40,
      fontFace: "Georgia", fontSize: 15, bold: true, color: NAVY, margin: 0,
    });
    s.addText(body, {
      x: 1.45, y: y + 0.30, w: 8.20, h: 0.65,
      fontFace: "Calibri", fontSize: 11.5, color: SLATE, margin: 0,
    });
  });

  footer(s, 9);
  s.addNotes(
    "Three positive findings that compounded into the v46 result.\n\n" +
    "One. Dataset size beat regularization on this small-N problem. We tried both levers. " +
    "Four textbook regularizers added +0.011 LB. Soft pseudo-labels expanding the effective " +
    "training set by 50,000 cells added +0.039. Three and a half times the gain from a different mechanism entirely.\n\n" +
    "Two. Within-recipe ensembling — same model, different seeds, SWA on the last few epochs — " +
    "added a clean small gain in v46. But cross-recipe sigmoid-averaging failed both times we tried it: " +
    "v22 with v19+v21 at a 0.05 LB gap, and v45_probe with v41+v44 at a 0.025 gap. Both regressed below " +
    "the stronger member. The operational rule is: don't ensemble across recipes unless members are within 0.02 LB.\n\n" +
    "Three. Patient identity is the dominant adversarial signal on this data. Every mitigation that worked — " +
    "patient-grouped MIL, AdaBN, test-stain norm, the patient_id=-1 sentinel for pseudo cells — " +
    "directly addresses the failure mode that broke v42."
  );
}

// =========================================================================
// Slide 10: NEGATIVE RESULTS GRID
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  dotMotif(s);
  eyebrow(s, "Negative results");
  title(s, "Six failures we diagnosed and fed forward");

  // 2x3 grid of failure cards
  const failures = [
    { x: 0.40, y: 1.55, head: "v22 ensemble collapse",     diag: "Cross-recipe avg at 0.05 LB gap → −0.003 below v19" },
    { x: 3.60, y: 1.55, head: "v27 val holdout failure",   diag: "2-patient holdout too noisy on N=12 → best_epoch=0" },
    { x: 6.80, y: 1.55, head: "v37/v38 Lian transfer",     diag: "FL aug + early fusion didn't transfer to 1-ch grayscale" },
    { x: 0.40, y: 3.05, head: "v42 SSL patient shortcut",  diag: "InfoNCE solved by patient ID → CV 0.96, LB 0.59" },
    { x: 3.60, y: 3.05, head: "v43 stacked regression",    diag: "4 simultaneous changes → −0.012, no attribution" },
    { x: 6.80, y: 3.05, head: "v45_probe cross-recipe",    diag: "0.025 LB gap still too wide → confirmed v22 rule" },
  ];

  failures.forEach(({ x, y, head, diag }) => {
    // Card
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 3.00, h: 1.35,
      fill: { color: WHITE }, line: { color: "E5E7EB", width: 0.5 },
      shadow: { type: "outer", color: "000000", opacity: 0.06, blur: 6, offset: 2, angle: 90 },
    });
    // Red accent stripe top
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 3.00, h: 0.06,
      fill: { color: CRIMSON }, line: { color: CRIMSON, width: 0 },
    });
    // Headline
    s.addText(head, {
      x: x + 0.18, y: y + 0.18, w: 2.65, h: 0.38,
      fontFace: "Georgia", fontSize: 12.5, bold: true, color: NAVY, margin: 0,
    });
    // Diagnosis
    s.addText(diag, {
      x: x + 0.18, y: y + 0.58, w: 2.65, h: 0.70,
      fontFace: "Calibri", fontSize: 10, color: SLATE, margin: 0,
    });
  });

  // Bottom takeaway
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.30, y: 4.65, w: 9.40, h: 0.42,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Every diagnosed failure shaped the v46 recipe directly. The wins are the losses processed.", {
    x: 0.50, y: 4.68, w: 9.20, h: 0.38,
    fontFace: "Calibri", fontSize: 11.5, italic: true, color: CREAM,
    align: "left", valign: "middle", margin: 0,
  });

  footer(s, 10);
  s.addNotes(
    "Six diagnosed failures. Each one fed forward into the v46 winning recipe.\n\n" +
    "v22: cross-recipe ensemble at 0.05 LB gap — regressed below v19. Rule encoded.\n\n" +
    "v27: tried a held-out 2-patient validation split. The selected patients had extreme FL exposure. " +
    "Best-epoch policy locked onto random init. Switched to public LB as our external validator.\n\n" +
    "v37/v38: tried to transfer Lian 2024's modality-specific FL aug and early fusion. Lian's setup is " +
    "4-channel emission stacks at higher resolution; ours is 1-channel grayscale at 128. Didn't transfer.\n\n" +
    "v42: the SSL collapse we discussed earlier. Patient shortcut.\n\n" +
    "v43: the four-stacked-changes regression. Confounded experiment.\n\n" +
    "v45_probe: cross-recipe ensemble at a tighter 0.025 gap. Still regressed. Rule confirmed.\n\n" +
    "Every one of these has a one-line rule that propagated forward. v46 is the cumulative response to " +
    "all six failures. The negative results are not separate from the methodology — they ARE the methodology."
  );
}

// =========================================================================
// Slide 11: FINAL SUBMISSION STRATEGY
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  dotMotif(s);
  eyebrow(s, "Strategy");
  title(s, "Two picks for the private leaderboard");

  // Two big cards
  statCard(s, 0.40, 1.55, 4.50, 1.85, "v46", "primary pick · LB 0.8236 · soft-pseudo distillation", EMERALD);
  statCard(s, 5.20, 1.55, 4.50, 1.85, "v41", "safety net · LB 0.7563 · no pseudo · maximally different recipe", NAVY);

  // Reasoning
  s.addText("Why these two — not v44 or v47", {
    x: 0.40, y: 3.65, w: 9.30, h: 0.30,
    fontFace: "Calibri", fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0,
  });
  s.addText([
    { text: "v44 shares pseudo-label lineage with v46 → correlated → no diversification", options: { bullet: true, breakLine: true } },
    { text: "v47 (noisy-student round 2) is even more correlated with v46 — same teacher chain", options: { bullet: true, breakLine: true } },
    { text: "v41 has no pseudo, no SWA, no FL aug — orthogonal failure mode → real hedge", options: { bullet: true } },
  ], {
    x: 0.40, y: 4.00, w: 9.30, h: 1.05,
    fontFace: "Calibri", fontSize: 11.5, color: SLATE,
    paraSpaceAfter: 5, margin: 0,
  });

  footer(s, 11);
  s.addNotes(
    "Kaggle lets us pick two submissions for the private leaderboard. Our plan:\n\n" +
    "Pick 1 is v46 — our LB 0.8236 result, currently #1 on the public board. Soft-pseudo distillation, " +
    "3 seeds × SWA, 40-way TTA. The headline.\n\n" +
    "Pick 2 is v41 — our LB 0.7563 result. The intuition: if v46 overfits to the public split or to v44's " +
    "pseudo distribution, we want a maximally-different safety net. v41 uses no pseudo-labels, no SWA, no " +
    "FL aug — it's a completely different mechanism. If the private LB shakes out badly for v46, v41 is the " +
    "best uncorrelated backup we have.\n\n" +
    "We deliberately do not pick v44 or v47 as the safety. Both share the pseudo-label lineage with v46. " +
    "A correlated pair offers no diversification. v41 is the lowest-correlation pick available."
  );
}

// =========================================================================
// Slide 12: TAKE-HOME MESSAGES (dark closing slide)
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // Teal accent bar on left
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.22, h: 5.625,
    fill: { color: TEAL }, line: { color: TEAL, width: 0 },
  });

  s.addText("FIVE THINGS WE'LL CARRY FORWARD", {
    x: 0.80, y: 0.50, w: 8.5, h: 0.30,
    fontFace: "Calibri", fontSize: 11.5, bold: true, color: TEAL, charSpacing: 5, margin: 0,
  });
  s.addText("Take-home messages", {
    x: 0.80, y: 0.85, w: 8.6, h: 0.65,
    fontFace: "Georgia", fontSize: 30, bold: true, color: WHITE, margin: 0,
  });

  const takeaways = [
    { head: "Higher CV ≠ higher LB.",
      body: "v15 had our best CV (0.866). It had our worst LB (0.572)." },
    { head: "Don't stack unvalidated changes.",
      body: "v43 stacked four tweaks at once → −0.012 LB and no way to attribute blame." },
    { head: "Seed variance is huge.",
      body: "Same recipe, different seed: 0.03 LB shift. A/B claims under 0.03 are noise." },
    { head: "On small-N, dataset size > regularization.",
      body: "v41's four regularizers gained +0.011 LB. v46's soft pseudo-labels gained +0.039." },
    { head: "Negative results are the methodology.",
      body: "Six diagnosed failure modes shaped the v46 recipe directly." },
  ];

  const y0 = 1.90;
  takeaways.forEach(({ head, body }, i) => {
    const y = y0 + i * 0.62;
    // Number
    s.addText(String(i + 1).padStart(2, "0"), {
      x: 0.80, y: y, w: 0.55, h: 0.40,
      fontFace: "Georgia", fontSize: 18, bold: true, color: AMBER,
      align: "left", margin: 0,
    });
    s.addText(head, {
      x: 1.40, y: y, w: 8.20, h: 0.32,
      fontFace: "Calibri", fontSize: 13, bold: true, color: WHITE, margin: 0,
    });
    s.addText(body, {
      x: 1.40, y: y + 0.32, w: 8.20, h: 0.30,
      fontFace: "Calibri", fontSize: 11, italic: true, color: CREAM, margin: 0,
    });
  });

  // Footer link
  s.addText("Code + LB history: github.com/rafallex/A3-ADL  ·  Questions?", {
    x: 0.80, y: 5.20, w: 8.6, h: 0.30,
    fontFace: "Calibri", fontSize: 10.5, italic: true, color: MUTED, margin: 0,
  });

  s.addNotes(
    "Five things to carry forward.\n\n" +
    "One. Higher CV doesn't mean higher LB. v15 had our best cross-validation result, 0.866, " +
    "and our worst leaderboard, 0.572. If you only have CV, you don't know what you have.\n\n" +
    "Two. Don't stack unvalidated changes. v43 added four tweaks at once on top of v41 and " +
    "regressed by 0.012 LB. The run cost 5 hours of GPU and produced no isolable signal. " +
    "We now change at most one or two related variables per submission.\n\n" +
    "Three. Seed variance is huge. v23 was v19 with only the random seed changed. 0.03 LB shift. " +
    "Most A/B comparisons with deltas under 0.03 are noise, not signal.\n\n" +
    "Four. On 12 patients, dataset size beats regularization. v41's four textbook regularizers " +
    "gained 0.011 LB. v46's soft pseudo-labels gained 0.039. Three and a half times the lift from " +
    "a different lever entirely.\n\n" +
    "Five. Negative results are the methodology. The v46 winning recipe is a direct response to " +
    "six diagnosed failures: v22, v27, v37, v38, v42, v43, and v45_probe. Reading the failures " +
    "faithfully is what produced the win.\n\n" +
    "Thanks. The full code and version-by-version LB history is at github.com/rafallex/A3-ADL. " +
    "Questions?"
  );
}

// === Save ===
const OUT = path.join(__dirname, "A3_cancer_challenge_claude.pptx");
pres.writeFile({ fileName: OUT }).then(() => {
  console.log("Saved:", OUT);
}).catch(err => {
  console.error("Build error:", err);
  process.exit(1);
});
