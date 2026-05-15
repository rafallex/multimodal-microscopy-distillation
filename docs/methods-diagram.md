# v16 methods — end-to-end diagram

GitHub renders Mermaid diagrams natively. Open this file on GitHub for the visual; the text below works as the source-of-truth for the README and report.

---

## Overall pipeline

```mermaid
flowchart TB
    subgraph Data["Data (12 train patients, ~10k cells each)"]
        BF["BF microscopy<br/>128x128 grayscale"]
        FL["FL microscopy<br/>128x128 grayscale"]
    end

    subgraph Stage1["Stage 1: CoMIR SSL pretraining (~25 min)"]
        SSL_aug["Heavy stain aug<br/>ColorJitter 0.5 + RandomGamma<br/>(independent per modality)"]
        SSL_model["Two ResNet-18 branches<br/>+ projection heads f_d -> 128"]
        SSL_loss["NT-Xent contrastive loss<br/>tau = 0.1"]
        SSL_aug --> SSL_model --> SSL_loss
    end

    subgraph Stage2["Stage 2: LOPO supervised (~4h 20min)"]
        LOPO["Leave-one-patient-out<br/>12 folds, single seed"]
        sup_model["Two ResNet-18 branches<br/>(init from CoMIR)<br/>+ frozen projection heads<br/>+ MLP head"]
        sup_loss["BCE + 0.05 x NT-Xent<br/>(aux alignment)"]
        EMA["EMA shadow (decay 0.99)<br/>updated each step"]
        snap["Snapshot at ep {3, 5}<br/>save EMA weights"]
        LOPO --> sup_model --> sup_loss
        sup_loss --> EMA --> snap
    end

    subgraph Stage3["Stage 3: Full-data x 2 seeds (~44 min)"]
        full["Same recipe as Stage 2<br/>but trains on all 12 patients<br/>seeds 107 and 207<br/>4 snapshots saved"]
    end

    subgraph Stage4["Stage 4: TTA inference + ensemble (~2h)"]
        tta["8-way D4 TTA per ckpt<br/>(4 rotations x 2 reflections)"]
        logits["Logit-space averaging<br/>(28 ckpts)"]
        mix["0.5 x sigmoid(mean_logit)<br/>+ 0.5 x rank_avg"]
        sub["submission.csv<br/>+ 3 diagnostic submissions"]
        tta --> logits --> mix --> sub
    end

    BF --> SSL_aug
    FL --> SSL_aug
    SSL_loss -.->|"saved backbone + projections"| sup_model
    snap --> tta
    full --> tta

    style Stage1 fill:#e3f2fd
    style Stage2 fill:#e8f5e9
    style Stage3 fill:#fff3e0
    style Stage4 fill:#fce4ec
```

---

## The model architecture

```mermaid
flowchart LR
    bf_in["BF image<br/>1 x 128 x 128"]
    fl_in["FL image<br/>1 x 128 x 128"]

    subgraph backbones["Backbones (init from CoMIR SSL)"]
        bf_branch["ResNet-18<br/>1-channel input<br/>fc replaced with Identity"]
        fl_branch["ResNet-18<br/>1-channel input<br/>fc replaced with Identity"]
    end

    bf_feat["BF features<br/>512-d"]
    fl_feat["FL features<br/>512-d"]

    subgraph proj["Projection heads (FROZEN, from CoMIR)"]
        bf_proj["Linear 512 -> 512 -> 128<br/>L2 normalized"]
        fl_proj["Linear 512 -> 512 -> 128<br/>L2 normalized"]
    end

    subgraph head["Classification head"]
        concat["Concat<br/>1024-d"]
        mlp["Linear 1024 -> 256<br/>BN -> ReLU -> Dropout(0.3)<br/>Linear 256 -> 1"]
    end

    bf_in --> bf_branch --> bf_feat
    fl_in --> fl_branch --> fl_feat
    bf_feat --> bf_proj
    fl_feat --> fl_proj
    bf_feat --> concat
    fl_feat --> concat
    concat --> mlp
    mlp --> logit["logit"]

    bf_proj --> z_bf["z_bf<br/>128-d unit"]
    fl_proj --> z_fl["z_fl<br/>128-d unit"]

    z_bf --> aux["NT-Xent aux loss<br/>x 0.05"]
    z_fl --> aux
    logit --> bce["BCE + label smoothing<br/>x 1.0"]

    aux --> total["total loss"]
    bce --> total

    style proj fill:#ffebee
    style head fill:#e8f5e9
    style aux fill:#fff3e0
```

Key points:
- The two branches are **independent** ResNet-18 networks (no weight sharing).
- The projection heads come from CoMIR SSL and stay **frozen** during supervised fine-tune. Gradient still flows *through* them to the backbones — they shape the alignment signal but don't drift away from their SSL solution.
- Inference uses only the `logit` path; the projection heads cost nothing at test time.

---

## Why this architecture?

```mermaid
flowchart TB
    problem["v15 problem:<br/>CV 0.866 -> LB 0.572<br/>OOD gap of 0.29"]

    cause1["Cause 1<br/>3-fold CV hid OOD failure"]
    cause2["Cause 2<br/>mixup fought CoMIR alignment"]
    cause3["Cause 3<br/>BCE drifted away from SSL features"]
    cause4["Cause 4<br/>best-AUC ckpt selection was<br/>inconsistent across folds"]

    fix1["Fix 1: LOPO 12-fold CV<br/>(change #2)"]
    fix2["Fix 2: no mixup<br/>(change #1)"]
    fix3["Fix 3: aux NT-Xent loss<br/>on frozen projections<br/>(change #9)"]
    fix4["Fix 4: multi-snapshot ensemble<br/>at ep {3,5} + EMA<br/>(changes #7, #8)"]

    problem --> cause1 --> fix1
    problem --> cause2 --> fix2
    problem --> cause3 --> fix3
    problem --> cause4 --> fix4

    fix1 --> v16["v16<br/>(plus changes #3-#6 for<br/>secondary robustness)"]
    fix2 --> v16
    fix3 --> v16
    fix4 --> v16

    style problem fill:#ffcdd2
    style v16 fill:#c8e6c9
```

---

## Compute timing on Kaggle T4×2 (only GPU 0 used)

```mermaid
gantt
    title v16 Kaggle commit timeline (~8h 15min)
    dateFormat HH:mm
    axisFormat %H:%M

    section Setup
    JPEG cache to RAM           :a1, 00:00, 45m

    section Stage 1: SSL
    CoMIR contrastive pretrain  :a2, after a1, 25m

    section Stage 2: LOPO
    12 folds x 6 epochs         :a3, after a2, 4h 20m

    section Stage 3: Full-data
    2 seeds x 6 epochs          :a4, after a3, 44m

    section Stage 4: Inference
    28 ckpts x 8-way D4 TTA     :a5, after a4, 2h

    section Buffer
    headroom under 9h limit     :a6, after a5, 45m
```

Total: **~8h 15min** of compute, leaving ~45 min of headroom under Kaggle's 9-hour commit ceiling. The cache step is the only one that has to run sequentially before anything else; all other stages depend on the previous one but use only GPU 0.

---

## If Mermaid doesn't render

GitHub renders Mermaid blocks automatically when the file is viewed in the web UI. If you're reading this in an editor that doesn't render Mermaid, install the [Mermaid VSCode extension](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) or paste the diagram source into <https://mermaid.live/>.
