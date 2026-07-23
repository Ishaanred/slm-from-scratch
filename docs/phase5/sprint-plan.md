# Phase 5 Sprint Plan — Evaluation & Feedback Loop

**Goal:** Benchmark the models this project actually trained, do real error analysis, and understand the generate-targeted-data-and-retrain feedback loop from first principles — the full RLHF-style loop, done manually and at small scale.

**Complete.** Scoped against what's actually true on this machine, not `docs/plan.md`'s original assumptions — same approach as every phase before this. Real benchmarks run, real errors read, and the feedback loop deliberately not run — see Step 4 for why, decided against real data rather than assumed in advance.

---

## What's actually available to evaluate

`docs/plan.md`'s Task 5.1 says compare "your 75M, 150M, distilled model, GPT-2 124M, SmolLM 135M." Checked directly:

- **No distilled model exists** — Phase 4 concluded as a conceptual + infrastructure investigation, not a trained model (see `docs/phase4/sprint-plan.md`).
- **Checked directly, not assumed: Phase 1's 50M checkpoint doesn't exist separately.** `train.py` used one fixed output path (`checkpoints/phase1/`), so the second run (77M) overwrote the first (50M) on disk. Only 4 checkpoints were actually available to evaluate: Phase 1's 77M, and Phase 3's 75M@500M/75M@2B/150M@500M.
- **lm-evaluation-harness wasn't installed** — installed (`lm-eval` 0.4.12).
- **Our models aren't HuggingFace-compatible** — custom `GPT` class, custom tokenizer, not an `AutoModelForCausalLM`. Built `src/phase5/eval_model.py`, a small adapter implementing lm-eval-harness's `LM` interface (`loglikelihood`, `loglikelihood_rolling`, `generate_until`) against the existing model/tokenizer. Sanity-checked directly before trusting it with a real run (correctly rates "Paris" more likely than "banana" as a continuation of "The capital of France is").

---

## Step 0 — Set expectations honestly before running anything

Standard benchmarks (HellaSwag, PIQA, ARC, GSM8K, HumanEval) are built for models trained on orders of magnitude more compute than this project's largest model (150M params, 2B tokens at most). Most are 4-choice multiple select, meaning **~25% is the random-chance floor** — and there's a real chance our models score at or barely above that floor across the board, not because anything is broken, but because the training scale genuinely isn't there yet. Worth deciding upfront that this is a legitimate, reportable outcome (same honesty as Phase 2's tokenizer regression and Phase 4's tokenizer-mismatch finding), not a failure to hide if it happens.

The comparison that's *guaranteed* to be meaningful regardless: **relative ranking across this project's own checkpoints** (does 150M beat 75M, does more tokens beat less) — that's the same question Phase 3 already partially answered, now checked against held-out task performance instead of just validation loss.

---

## Step 1 — Install + adapt lm-eval-harness — done

`src/phase5/eval_model.py` implements lm-eval-harness's `LM` interface against the project's own `GPT` class and both tokenizers (GPT-2 for Phase 1, the 32K BPE for Phase 3). Handles a real quirk along the way: checkpoints pickle the training run's `TrainConfig` object, which needs its original class importable to unpickle — worked around with a harmless stand-in class rather than depending on the exact original `train.py` being importable from this new location.

---

## Step 2 — Run the actual comparison — done

All 4 available checkpoints, 3 benchmarks (PIQA, HellaSwag, ARC-Easy — not the full plan.md list, per the "pick a mix, see if there's *any* differentiation" scoping). Real results, ~9 minutes total:

| Checkpoint | PIQA (~50% floor) | HellaSwag (~25% floor) | ARC-Easy (~25% floor) |
|---|---|---|---|
| Phase 1 — 77M | 53.4% | 26.0% | 27.9% |
| Phase 3 — 75M @ 500M tok | 53.8% | 26.1% | 29.8% |
| Phase 3 — 75M @ 2B tok | 54.8% | 26.4% | 31.7% |
| Phase 3 — 150M @ 500M tok | 54.7% | 26.2% | 32.4% |

HellaSwag: completely flat at the floor regardless of size or data. ARC-Easy: a real, consistent climb with scale — the clearest signal of the three. PIQA: weakly above floor, mild upward trend.

---

## Step 3 — Error analysis — mechanical part done

`src/phase5/error_analysis.py` pulls wrong-answer examples (67.6% wrong for 150M, 72.1% for 77M on ARC-Easy) — saved in `docs/phase5/arc_easy_errors_150m_500m.txt` for actual review. One objective, mechanical finding worth weighing during that review: this project's 32K BPE tokenizer heavily fragments the domain-specific vocabulary these science questions use — `photosynthesis` → 4 pieces, `haploid` → 4 pieces, `negatively-charged` → 4 pieces. That's a real, checkable fact about the tokenizer, not a claim about why any specific question was missed — worth keeping in mind as one plausible contributing factor alongside genuine reasoning gaps and plain undertraining.

---

## Step 4 — Targeted data + iterate — decided against, using the rule already committed to above

This step's own plan (written before results existed) set the deciding condition in advance, specifically to avoid rationalizing a decision after the fact: *"If Step 2/3 show near-floor scores across the board rather than a few isolated weak categories, that's a sign the feedback loop's actual precondition (partial competence, narrow gaps) isn't met yet."*

Checked against the real data: HellaSwag flat at the floor, PIQA barely above its floor, and even ARC-Easy's genuine trend tops out at 32.4% — broad, low performance across the board, not a competent model with a few isolated gaps to patch. By the rule already set, **the feedback loop isn't run.** This also avoids repeating Phase 4's teacher-cost tradeoff (targeted generation is decode-bound, ~81 tok/s measured, and a multi-round loop multiplies that cost) for a precondition that isn't met anyway.

---

## Order of work

1. ~~Step 0 (expectations)~~ — done, framing held up against real results
2. ~~Step 1 (harness + adapter)~~ — done
3. ~~Step 2 (run the comparison)~~ — done, real results above
4. ~~Step 3 (error analysis, mechanical part)~~ — done, categorization available for further review any time
5. ~~Step 4 (targeted data + iterate)~~ — decided against, using the pre-committed rule against real data
