# Phase 5 Sprint Plan — Evaluation & Feedback Loop

**Goal:** Benchmark the models this project actually trained, do real error analysis, and understand the generate-targeted-data-and-retrain feedback loop from first principles — the full RLHF-style loop, done manually and at small scale.

**Not started yet.** This plan is scoped against what's actually true on this machine, not `docs/plan.md`'s original assumptions — same approach as every phase before this.

---

## What's actually available to evaluate

`docs/plan.md`'s Task 5.1 says compare "your 75M, 150M, distilled model, GPT-2 124M, SmolLM 135M." Checked directly:

- **No distilled model exists** — Phase 4 concluded as a conceptual + infrastructure investigation, not a trained model (see `docs/phase4/sprint-plan.md`).
- **Real checkpoints that do exist**: Phase 1's 50M and 77M (GPT-2 tokenizer, ~80M tokens), Phase 3's 75M @ 500M tokens, 75M @ 2B tokens, and 150M @ 500M tokens (32K BPE tokenizer, full corpus).
- **lm-evaluation-harness isn't installed** (`pip index versions lm-eval` shows 0.4.12 available, not yet pulled in).
- **Our models aren't HuggingFace-compatible** — custom `GPT` class, custom tokenizer, not an `AutoModelForCausalLM`. lm-eval-harness needs a small custom adapter (subclassing its `LM` interface, implementing `loglikelihood`) to evaluate them at all — this is infrastructure, not a re-architecture.

---

## Step 0 — Set expectations honestly before running anything

Standard benchmarks (HellaSwag, PIQA, ARC, GSM8K, HumanEval) are built for models trained on orders of magnitude more compute than this project's largest model (150M params, 2B tokens at most). Most are 4-choice multiple select, meaning **~25% is the random-chance floor** — and there's a real chance our models score at or barely above that floor across the board, not because anything is broken, but because the training scale genuinely isn't there yet. Worth deciding upfront that this is a legitimate, reportable outcome (same honesty as Phase 2's tokenizer regression and Phase 4's tokenizer-mismatch finding), not a failure to hide if it happens.

The comparison that's *guaranteed* to be meaningful regardless: **relative ranking across this project's own checkpoints** (does 150M beat 75M, does more tokens beat less) — that's the same question Phase 3 already partially answered, now checked against held-out task performance instead of just validation loss.

---

## Step 1 — Install + adapt lm-eval-harness

- `pip install lm-eval`
- Write a thin adapter (`src/phase5/eval_model.py` or similar) wrapping this project's `GPT` + BPE tokenizer to satisfy lm-eval-harness's `LM` interface (`loglikelihood`, `loglikelihood_rolling` at minimum — most multiple-choice benchmarks only need `loglikelihood`). This is infrastructure/plumbing, not a learning-objective piece — happy to write it.
- Pick a **small, cheap benchmark subset first** to sanity-check the adapter works at all (e.g., a few hundred examples of one task) before running the full suite across every checkpoint.

---

## Step 2 — Run the actual comparison

- All 5 of this project's own checkpoints (Phase 1's 50M/77M, Phase 3's 75M@500M/75M@2B/150M@500M)
- Optionally GPT-2-124M and/or SmolLM-135M pulled from HuggingFace as external reference points, if the adapter effort for our own models is working — these are already lm-eval-compatible off the shelf, no extra adapter work needed
- 2-4 benchmarks, not the full plan.md list — pick a mix (one commonsense like PIQA, one that needs more reasoning like ARC-Challenge or GSM8K) to see if there's *any* differentiation, rather than running the full 6-benchmark suite blind

---

## Step 3 — Error analysis

- Pull the worst-scoring examples per benchmark per model (lm-eval-harness supports per-example output logging)
- Categorize by hand: is it tokenization (words split oddly by the 32K BPE tokenizer), genuine reasoning failure, or just "too small to have learned this at all"
- This step doesn't need the teacher at all — it's reading the model's own outputs against the real answers, real analysis work

---

## Step 4 — Targeted data + iterate — scope this against Phase 4's lesson, not blindly

`docs/plan.md`'s Tasks 5.3-5.4 call for generating 5K-20K targeted synthetic examples per error category via the teacher, fine-tuning, and repeating. Worth being honest about the same tradeoff Phase 4 surfaced: the teacher's *generation* throughput is decode-bound (~81 tok/s measured in Phase 4), not the fast prefill regime — generating meaningful volumes of targeted data costs real time and electricity per round, and a multi-round iterate loop multiplies that cost.

There's also a conceptual mismatch worth checking before committing: **targeted synthetic data fixes specific, isolated weak spots in an otherwise-competent model** — it's not a tool for fixing "the model is too small and undertrained to have learned this category at all." If Step 2/3 show near-floor scores across the board rather than a few isolated weak categories, that's a sign the feedback loop's actual precondition (partial competence, narrow gaps) isn't met yet, and a full iterate loop may not be the right next move regardless of teacher cost. Decide this **after** seeing Step 2/3's real results, not before.

---

## Order of work

1. Step 0 (expectations) — no cost, just framing
2. Step 1 (harness + adapter) — infrastructure, build this first
3. Step 2 (run the comparison) — the first real data point
4. Step 3 (error analysis) — cheap, no teacher needed
5. Step 4 (targeted data + iterate) — revisit scope once Step 2/3's actual results are in, same measure-before-committing approach as every phase before this
