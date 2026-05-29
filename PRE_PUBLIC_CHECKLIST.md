# Pre-Public Release Checklist

Internal note — **delete this file as the last step before flipping the repo to public.**
Tracks what must be true before `rafallex/multimodal-microscopy-distillation` goes from
private → public. Audited 2026-05-29.

## ⚠️ READ FIRST — do not just click "Make public"
The repo's *files* are clean, but **old git commits still contain 27 classmates' Kaggle
usernames** (see the scrub section). If you flip to public without scrubbing history,
that data becomes public. So the one-line rule:

> **When the course is over, message Claude: "scrub history and publish."**
> Claude runs the history scrub, verifies it's clean, force-pushes, and tells you when it's
> safe to click Settings → Make public. Everything else in this file is cosmetic.

If you'd rather do it yourself, follow the scrub procedure below step by step.

## ✅ Already clean (verified)
- **No secrets / API keys / tokens** in tracked files (`.gitignore:220` is just an *ignore rule* for `.streamlit/secrets.toml`, not a secret).
- **No hard-coded personal paths** (`C:\Users\Rafallex\...`) in tracked files.
- **No other-student usernames in the current tree** (redacted in commit `ed38ea3`).
- **Dataset is gitignored** (`multimodal-cancer-classification-challenge-2026/`, BF/FL/train.csv never tracked).
- **LICENSE** present (MIT).
- **README** is portfolio-grade; repo renamed; description fixed.
- **Notebook outputs stripped** from `notebooks/legacy/*.ipynb` (this branch). Current source notebooks carry no outputs.
- Author email (`rafaeltproenca@gmail.com`) appears only in `overleaf-report/main.tex` as the paper author line — intentional, your own address. Remove it there if you'd rather not have it public.

## ⛔ Gated preconditions — do NOT go public until ALL are true
1. **Competition closed** — Kaggle comp ends **2026-06-03 21:59 UTC**. Publishing earlier lets other teams clone your `results/*.csv` predictions and resubmit/ensemble them against you.
2. **Grade received (recommended)** — wait until the A3 module is graded, in case of any academic-integrity review of public code.
3. **All work committed/merged to `main`** — the history scrub below rewrites every commit SHA, so do it only after the last commit is in.

## 🔴 The one real blocker: scrub other-student usernames from git HISTORY
27 classmates' Kaggle handles were committed in the leaderboard table between commits
`7444718` and `ed38ea3` (added in `7444718`, removed in `ed38ea3`). They are gone from
the current files but still visible in the diff/history of that range. Under GDPR this
is identifiable data on third parties we have no consent to publish.

### Procedure (run only at the end, after the gated preconditions)
```bash
# 0. fresh clone to operate on (safer than rewriting your working copy)
git clone https://github.com/rafallex/multimodal-microscopy-distillation.git scrub && cd scrub

# 1. build a transient replace-rules file OUTSIDE the repo (never committed).
#    Open the leaked table and copy each handle from the "Members" column:
git show 7444718:LB_HISTORY.md | sed -n '/Top-of-leaderboard/,/^We were briefly/p'
#    For every handle you see, add one line to /tmp/scrub-rules.txt in this exact form:
#        thehandle==>***
#    (one per username; ~27 lines). Do NOT save this file inside the repo.

# 2. rewrite history, replacing every occurrence across all commits
python -m git_filter_repo --replace-text /tmp/scrub-rules.txt --force

# 3. filter-repo drops the remote for safety — re-add and force-push the rewritten history
git remote add origin https://github.com/rafallex/multimodal-microscopy-distillation.git
git push --force --all origin
git push --force --tags origin

# 4. VERIFY the history is clean (pick ANY one handle from step 1; must print nothing):
git log --all -S "<paste-one-handle-here>" --oneline
git -c grep.lineNumber=false grep -iI "<paste-one-handle-here>" $(git rev-list --all) 2>/dev/null

# 5. rm /tmp/scrub-rules.txt   (delete the transient PII file)
```
> Easier option: when you're ready, just ask Claude to run this — the handles can be
> re-derived from commit `7444718` at that time, so you don't have to transcribe them.

## 🟡 Optional polish (nice-to-have, not blockers)
- **Final LB update** — once the private leaderboard is announced, drop the final
  standing into the README's opening (it's currently phrased to age gracefully, so this
  is optional, not required).
- `notebooks/legacy/` — early exploratory notebooks (v2–v16). Outputs are stripped; keep
  them (shows the real iteration journey) or `git rm` them for a tighter portfolio.
- ~~Add a `CITATION.cff`~~ — done.
- Tag the paper-submission commit (e.g. `git tag v1.0-paper`).
- **Commit attribution (your call):** most commits carry a `Co-Authored-By: Claude`
  trailer (honest — Claude helped with repo organization, the paper, and analysis
  scripts; the ML experiments and decisions were yours). AI-assisted work is mainstream
  and keeping it is the straightforward, honest choice. If you'd prefer the history to
  read as solely yours, the trailers can be dropped during the same `git-filter-repo`
  pass as the username scrub — tell Claude and it's one extra flag.

## Final flip
1. Confirm gated preconditions ✔, run the history scrub ✔, verify clean ✔.
2. `git rm PRE_PUBLIC_CHECKLIST.md && git commit -m "chore: drop internal pre-public checklist" && git push`
3. GitHub → repo **Settings → General → Danger Zone → Change visibility → Public**.
