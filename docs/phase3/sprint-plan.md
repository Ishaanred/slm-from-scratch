# Phase 3 Sprint Plan — Scaling Law Experiments

**Goal:** Run a small, deliberately chosen set of training runs to check where your own data lands relative to Chinchilla's compute-optimal formula (tokens ≈ 20 × params), and to pick a sensible model size for Phase 4's distillation student.

**Scope note (2026-07-14):** started as a 10-run grid (~80-120 GPU hours), trimmed to 6 (~55-70 hours), trimmed again to 3 runs (~24-25 hours). The harness reuses one fixed LR/config across all model sizes rather than retuning per size, so a bigger grid was never going to produce a textbook-clean scaling curve anyway — 3 well-chosen runs answering two specific questions (does more data help; does more parameters help when data is scarce) carries the same practical and portfolio value at a fraction of the cost. See `docs/plan.md`'s Phase 3 section for the exact 3 runs and reasoning. Not a today task either way — still ~1 GPU-day, unattended.

---

## Step 0 — Carried over from Phase 2

Re-tokenize the **full** OpenWebText corpus (~8.5B tokens) with the 32K BPE tokenizer, instead of the small ~214M-token sample Phase 2 used. The full-corpus version of `data/phase2/prepare.py` exists in git history (see the comment at the top of that file) — reviving it, pointed at `data/openwebtext/` instead of `data/phase2/clean.txt`, is most of this step.

This does double duty: it's required for Phase 3's larger token counts anyway, and it gives a clean, corpus-matched Phase 1 vs Phase 2 tokenizer comparison that Phase 2 couldn't produce on its own (see `docs/phase2/results.md`).

---

## Step 1 — Parameterize the training harness

Right now `src/phase2/train.py` has a fixed `TrainConfig` — one model size, one token count, hardcoded. Phase 3 needs the same architecture at multiple sizes and multiple step counts without copy-pasting a new file per run.

Needed:
- CLI args (or a config file) for `n_layer`, `n_head`, `n_embd`, `max_iters` — everything else in `TrainConfig` can stay fixed
- A distinct `wandb_run_name` / `out_dir` per run so results don't overwrite each other
- Somewhere to log each run's final params/tokens/loss so Step 3's plotting has a table to read from, rather than digging through W&B by hand

This is plumbing (config parsing, argparse), not a learning-objective piece — happy to write this when you're ready.

---

## Step 2 — Pick concrete configs

Only two sizes are used now: 75M and 150M. Measured directly (2026-07-13) by instantiating each config from `src/phase2/model.py` and running one real forward+backward pass on the 5070 Ti at batch=8, block_size=1024:

| Target | n_layer | n_head | n_embd | Actual params | Peak VRAM (batch=8, block=1024) |
|---|---|---|---|---|---|
| 75M | 8 | 8 | 576 | 69.4M | 4.94 GB |
| 150M | 12 | 12 | 768 | 135.0M | 6.70 GB |

Both fit on the 5070 Ti (16.3GB total, 14.5GB free) with plenty of room — VRAM was never the constraint, wall-clock time is. Token count doesn't affect VRAM at all, only run duration and disk usage for the tokenized corpus (full re-tokenization is ~17GB, disk has 732GB free).

The embedding dims above undershoot the round-number labels slightly (69.4M vs "75M", 135.0M vs "150M") — close enough to not matter.

Token counts translate to `max_iters` via `max_iters = tokens / (batch_size × block_size)` at whatever batch size the harness ends up using.

---

## Step 3 — Run, then plot

Once Steps 0-2 are done, this is where `docs/plan.md`'s Task 3.2 (run the experiments) and Task 3.3 (plot loss vs FLOPs, loss vs tokens, identify regimes) actually happen. Runs are fire-and-forget — start one, come back hours later.

---

## Order of work

1. Step 0 (full-corpus re-tokenization) — can start any time, doesn't block anything else
2. Step 1 (harness) — needed before any grid run
3. Step 2 (confirm configs) — quick, do right before the first real run
4. Step 3 (run + plot) — the bulk of the calendar time, spread across days
