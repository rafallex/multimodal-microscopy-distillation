# Speaker script (10 minutes)

Notes for delivery:
- First person, about 9 minutes 40 seconds, which leaves a little buffer under the 10-minute cap.
- Lead with method. The grade is on method and presentation, so the score is supporting evidence, not the point.
- For a team, hand off at the act boundaries: slides 1-4, 5-7, 8-10, 11-14.
- Take your time on slides 5, 8, and 12 (the three turning points). Move quickly through 3, 6, and 10.
- Don't read the slides aloud. Say the story and let the numbers sit on screen.

| # | Slide | Budget |
|---|---|---|
| 1 | Title | 0:25 |
| 2 | §1 Challenge | 0:45 |
| 3 | §2 Method | 0:45 |
| 4 | §3 The Arc | 0:35 |
| 5 | §4 SSL collapse | 0:50 |
| 6 | §5 Pivot (v19) | 0:40 |
| 7 | §6 v41 vs v43 | 0:40 |
| 8 | §7 Breakthrough | 0:50 |
| 9 | §7 Hard vs Soft | 0:45 |
| 10 | §8 Evidence | 0:30 |
| 11 | §9 Negative results | 0:40 |
| 12 | §10 Ensembling failed | 0:45 |
| 13 | §11 Final push | 0:45 |
| 14 | §12 Take-home | 0:45 |
| | **Total** | **≈ 9:40** |

---

### Slide 1 — Title (0:25)
"Hi, I'm Rafael. This is my run at the Multimodal Cancer Cell Classification challenge — thirty-plus logged versions. My best single model is 0.8392. But the score isn't the story: almost every gain came from reading a *failure* correctly, so that's what I'll focus on."

### Slide 2 — §1 The Challenge (0:45)
"The task is per-cell: is this oral cell from a cancer patient — from paired brightfield and fluorescence crops, 128 pixels, grayscale. Two facts shape everything. First, only **twelve** training patients, so labels are effectively patient-level and the real sample size is tiny. Second, the test patients are completely **disjoint** from training — out-of-distribution generalisation is the dominant failure mode. And the instructor was explicit: grading is on *what is done and how it's presented*, not the leaderboard. So that's how I built this talk."

### Slide 3 — §2 Method (0:45)
"The backbone is deliberately boring — two EfficientNet-B0 branches, one per modality, late-concatenated, with an MLP head. The interesting parts are everywhere *except* the backbone. A per-patient multiple-instance loss stops the model memorising within-patient quirks. AdaBN and test-set stain normalisation explicitly model the test distribution — the fluorescence test set is twenty percent brighter than train. Plus eight-way test-time augmentation. The recurring lesson: here, the architecture is not the lever."

### Slide 4 — §3 The Arc (0:35)
"The whole journey in five acts: baselines and self-supervision — which collapsed; the EfficientNet pivot that set a floor; a regulariser search; and then the semi-supervised breakthrough — pseudo-labels and distillation. The single biggest jump, +0.039, came from one idea: soft pseudo-labels. Let me walk the turning points."

### Slide 5 — §4 The Ambition → SSL collapse (0:50)
"Early on I tried cross-modal self-supervised pretraining. It gave the best cross-validation I'd ever seen — 0.866. I submitted expecting 0.75, and got **0.572** — basically random. The diagnosis took days: because every cell has exactly one patient, the contrastive task can be solved by encoding *patient identity* rather than cell content. The model found a shortcut. That's take-home number one: high CV does **not** mean high leaderboard — especially on patient-grouped, out-of-distribution data."

### Slide 6 — §5 The Pivot (0:40)
"So I dropped self-supervision and did the robust thing: model the test distribution directly. Five standard ingredients — dual EfficientNet, the MIL loss, AdaBN, stain norm, strong augmentation and TTA. No new architecture. That set the supervised floor at **0.7455** and held for three weeks. The win wasn't cleverness — it was treating the test distribution as the thing to model."

### Slide 7 — §6 The Search (0:40)
"Then a discipline lesson. v41 added four textbook regularisers, cleanly, for +0.011. v43 added four *more* at once — and regressed by 0.012, with no way to tell which change was the culprit. Five hours of GPU, no signal. The rule I encoded: change one or two related things per submission. Later, v46 reverted exactly the two suspects and recovered +0.079."

### Slide 8 — §7 The Breakthrough (0:50)
"This was the turning point: use the test set itself as a training signal. A teacher model predicts on the 59,000 unlabelled test cells, and the student trains on those predictions alongside the real labels. Two flavours. Hard pseudo-labels — keep only confident cells, assign zero or one — gave +0.025. But that throws away 84% of the test set. So I switched to Hinton-style **soft** distillation: keep *every* cell, train on the teacher's raw probability. That's the heart of the method — next slide."

### Slide 9 — §7 Hard vs Soft (0:45)
"Same teacher, same predictions — two ways to use them. Hard keeps sixteen percent of cells; soft keeps a hundred percent, each weighted by the teacher's confidence. The histogram shows why: the 84% in the middle — the uncertain cells — is exactly Hinton's 'dark knowledge', the information a hard zero-or-one throws away. The move to soft was **+0.039**, the single biggest jump in the project. A second distillation round added a little more and tightened the seed variance."

### Slide 10 — §8 Evidence (0:30)
"The whole story in one chart. The floor at 0.7455; the self-supervision collapse down here; the stacked-change regression; the pseudo-label climb to the top; and at the right edge, the final architecture push to my best, 0.8392. Green is breakthrough, red is regression — read it once and the narrative is the shape of the line."

### Slide 11 — §9 Negative Results (0:40)
"Six diagnosed failures — and I'd argue these *are* the methodology. Cross-recipe ensembles that collapsed. A held-out validation too noisy on twelve patients. Borrowed augmentation from the source paper that didn't transfer to grayscale. The self-supervision shortcut. The stacked-regulariser regression. Each produced a one-line rule that fed directly into the winning recipe. The wins are the losses, processed."

### Slide 12 — §10 Ensembling Failed (0:45)
"One finding worth highlighting: on this data, ensembling never gave a lift beyond noise. My three-seed average scored below my best single seed. Cross-recipe averages regressed. Even a genuinely decorrelated gradient-boosted model dragged me down, because its cross-validation didn't transfer. The one exception was a blend of my two best, near-tied models — it edged the best single by about a thousandth of a point, which is inside the noise floor. So with twelve patients a careful blend and the best single model are interchangeable. For my final lock-in I chose robustness over the lucky draw — two variance-reduced models, neither dependent on a single seed."

### Slide 13 — §11 The Final Push (0:45)
"Finally, architecture. I swept the remaining levers as single models. More capacity — EfficientNetV2-S — regressed to 0.74. Higher resolution, 192 pixels — regressed to 0.78. Both confirm the model is **data-bound**, not capacity-bound. The one thing that helped was fusion design: intermediate co-attention — the design from the source-dataset paper — reached a new best, **0.8392**. The learning curves on the left show clean, stable training; I had to harden the attention against a half-precision overflow to get there."

### Slide 14 — §12 Take-Home (0:45)
"Five things I'll carry forward. High CV doesn't mean high leaderboard. Don't stack unvalidated changes. Seed variance is huge — treat small deltas as noise. On small data, growing the *effective* dataset beats architecture by an order of magnitude. And negative results are the methodology — reading failures faithfully is what produced every win. Code and the full version history are on GitHub. Thank you — happy to take questions."

---

Rehearse once with a timer. If you run long, the easiest cuts are slides 7 and 10 (one sentence each), which buys about 40 seconds without losing the thread.
