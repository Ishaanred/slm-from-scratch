# SLM Training from Scratch — 2-Month Learning Roadmap

> **Goal:** Train a tiny language model (75M→500M params) from scratch on a single RTX 5070 Ti, learning every step of the pipeline: data engineering, tokenization, scaling laws, distillation, evaluation, and feedback loops.

**Hardware:** RTX 5070 Ti (16GB VRAM)
**Time budget:** ~2 months, 4-6 hrs/day (~200-300 hours total)
**Style:** Practical — use frameworks (PyTorch, HuggingFace), not from-scratch attention math. AI-assisted (you + Claude/Cursor doing the heavy coding while you learn the concepts).
**Teacher models:** Qwen 3.6 35B MoE (runs locally 4-bit), DeepSeek Flash v4 (API), RunPod 5090 (fallback for heavy distillation).

**Parallel project:** FrameWeaver — plan accounts for split attention. Most training runs are fire-and-forget (start training, check back in 2-8 hours).

---

## Architecture Overview

```
Phase 1: NanoGPT warm-up          →  Week 1       (~20 hrs)
Phase 2: Tokenizer + Data Eng     →  Week 2-3     (~40 hrs)  
Phase 3: Scaling law experiments  →  Week 3-5     (~50 hrs, much GPU-idle time)
Phase 4: Distillation             →  Week 5-7     (~50 hrs, much GPU-idle time)
Phase 5: Evaluation + Feedback    →  Week 7-8     (~40 hrs)
```

**Key insight:** Phases 3 and 4 are GPU-bound, not you-bound. Start a training run, work on FrameWeaver for 4-6 hours, come back to analyze results. This is how real ML engineers work.

---

## Phase 1: NanoGPT Warm-Up — "I Understand Transformers Now" ✓ COMPLETE

**Objective:** Build and train a minimal GPT from scratch (~300 lines of PyTorch). This demystifies every line of a transformer — attention, MLP blocks, positional embeddings, loss computation.

### Completed
- [x] Environment setup — PyTorch 2.6, CUDA 13.2, W&B
- [x] `model.py` — CausalSelfAttention (Flash Attention), MLP, Block, GPT written from scratch
- [x] `train.py` — full training loop: bfloat16, gradient accumulation, cosine LR, checkpointing
- [x] `generate.py` — top-k sampling from checkpoint
- [x] OpenWebText tokenized — 17GB train.bin (~8.5B tokens)
- [x] Run 1: 50M params, 5K steps, val loss 5.12
- [x] Run 2: 77M params, 5K steps, val loss 5.24 (undertrained — expected)
- [x] Visual explainer doc — `docs/phase1/how-transformers-work.html`
- [x] Results documented — `docs/phase1/results.md`

**Key finding:** At 80M tokens, the smaller 50M model outperforms the 77M. More parameters need more data. Phase 3 is where longer runs will show the 77M pulling ahead.

---

## Phase 2: Tokenizer + Data Engineering — "Garbage In, Garbage Out" ✓ COMPLETE

**Objective:** Train your own tokenizer and build a data pipeline. This is 60% of the work in real ML.

### Task 2.1: Train a BPE tokenizer — done
- [x] Trained BPE tokenizers at 8K/16K/32K on filtered OpenWebText sample (`src/phase2/train_tokenizer.py`)
- [x] Compared compression vs GPT-2 tokenizer on held-out text (`src/phase2/evaluate_tokenizer.py`) — see `docs/phase2/results.md`
- [x] Picked 32K vocab, re-tokenized the clean corpus

### Task 2.2: Data filtering pipeline — done
- [x] fastText language-ID (English only, confidence > 0.65)
- [x] Quality heuristics (length, symbol/digit ratio)
- [x] Exact dedup (content hash)
- **Reviewed and confirmed out of scope (2026-07-13):** PII removal and synthetic-data generation teach nothing specific to Phase 2's data-engineering goal — PII removal is a compliance concern, synthetic data via a teacher model is Phase 4's actual lesson. Near-dedup (MinHash LSH) is the one with real unexplored conceptual content (approximate similarity at scale, distinct from exact-hash dedup) but isn't blocking; revisit only if useful later.

**Key finding:** the 32K tokenizer trained here is marginally *worse* than GPT-2's at compression on held-out web text (-1.2%) — that's a clean, standalone result. The training run's bits-per-byte also came out slightly worse (1.736 vs 1.718), but that comparison is confounded: Phase 2's `train.bin` was built from a ~214M-token sample (scoped for tokenizer training, not LM training) vs Phase 1's full ~8.5B-token corpus, a ~40x difference in training data that alone could explain the gap. Don't read the training-run number as evidence the tokenizer hurt learning — only the tokenizer-only compression comparison supports that. Full breakdown in `docs/phase2/results.md`.

**Note on process:** Task 2.1's scripts (tokenizer training, evaluation) and Task 2.2's filter script were written by Claude in a prior session rather than hand-written, which is against this file's own learning-boundary rule for tokenizer training. Concepts were reviewed and understood afterward, but flagging this here for accuracy.

### Task 2.3: Synthetic data generation with teacher model — descoped to Phase 4
- **Teacher option 1 (local):** Qwen 3.6 35B MoE — runs 4-bit on your 5070 Ti. Use when you're not training.
- **Teacher option 2 (API):** DeepSeek Flash v4 — fast, cheap, good quality. Use when GPU is busy training.
- **Teacher option 3 (cloud):** RunPod 5090 for heavy batch generation (~10 hrs, borrow when needed)
- Generate:
  - **Instruction-following pairs:** "Write a function that..." → code
  - **Chain-of-thought reasoning:** "Solve this step by step..." → reasoning traces
  - **Text continuations:** Seed with high-quality text, let teacher continue
- Aim for 50K-200K high-quality synthetic samples

### Task 2.4: Dataset mixing — descoped to Phase 4
Depends on synthetic data from Task 2.3, so it moves with it. Phase 2's deliverable is the filtered web-text corpus alone; the 80/15/5 mix happens once synthetic + curated data exist.

**Deliverable — met:** a tokenizer you trained (32K BPE) + a clean, deduplicated dataset tokenized with it, verified end-to-end with a training run. See `docs/phase2/results.md`.

---

## Phase 3: Scaling Law Experiments — "Chinchilla Was Right"

**Objective:** Run controlled experiments to understand model size vs data size vs loss.

**Carried over from Phase 2:** re-tokenize the full OpenWebText corpus with the 32K BPE tokenizer (Phase 2 only tokenized a small ~214M-token sample). This phase's runs need full-corpus token counts anyway, so this becomes step 1 of data prep here rather than a separate task — and it incidentally gives a clean, corpus-matched Phase 1 vs Phase 2 tokenizer comparison (see `docs/phase2/results.md` for why the Phase 2 result wasn't conclusive on its own).

### The Experiment — trimmed to 3 runs (2026-07-14)

Original plan was a 10-run grid (~80-120 GPU hours), then trimmed to 6 (~55-70 hours). Trimmed again to 3 runs: given the harness reuses one fixed LR/config across all model sizes instead of retuning per size, and the corpus is still modest, a 6-10 run "clean scaling law curve" was never going to read as a textbook-quality Chinchilla reproduction anyway — a small, well-chosen set of runs that answers two specific questions carries the same practical and portfolio value at a fraction of the cost.

| Run | Model | Tokens | Tokens/param | What it answers |
|---|---|---|---|---|
| 1 | 75M | 500M | ~7.2x | Undertrained end of the 75M curve |
| 2 | 75M | 2B | ~28.8x | Near/above Chinchilla-optimal end of the 75M curve |
| 3 | 150M | 500M | ~3.7x | Cross-size comparison at matched (scarce) data — direct callback to Phase 1's finding that a bigger model can lose to a smaller one when data is scarce |

Runs 1 and 2 show the within-size trend (does more data help). Runs 1 and 3 show the cross-size trend at fixed data (does more parameters help when data is scarce). Two axes, three runs.

**Rough GPU time:** Run 1 ~3.5h, Run 2 ~14h, Run 3 ~7h — **~24-25 hours total**, roughly 2-2.5 days of continuous unattended wall-clock time.

### Task 3.1: Build the training harness
- Mixed precision (bfloat16), gradient accumulation, cosine LR with warmup, gradient clipping
- W&B logging: loss, perplexity, LR, tokens/sec
- Checkpoint resume
- `torch.compile()` + Flash Attention via PyTorch SDPA

### Task 3.2: Run the experiments
- Queue all 3 runs to launch back-to-back and let them run unattended — ~24-25 GPU hours, ~2-2.5 days continuous.
- Pro tip: Start the queue, switch to FrameWeaver, check W&B from your phone

### Task 3.3: Plot and analyze
- **Loss vs FLOPs** — color by model size. Chinchilla: compute-optimal tokens ≈ 20× params
- **Loss vs tokens** — identify undertrained vs overtrained regimes
- **Extrapolate:** If you had 10× compute, what's optimal?
- Write up findings — portfolio material

### Task 3.4: Optimization experiments (optional)
- Compare AdamW vs Adam vs SGD with momentum
- Compare LR schedules (cosine vs linear vs constant with cooldown)
- Compare batch sizes (32 vs 128 vs 512 via gradient accumulation)

**Deliverable:** Scaling law plots with your data points. Intuition for "how much data does my model need?"

---

## Phase 4: Knowledge Distillation — "The Student Surpasses the Master" ✓ COMPLETE

**Objective:** Understand knowledge distillation — the concepts, and what actually running it requires in practice.

**What this phase covers:** hard vs. soft labels, temperature scaling, KL divergence, and the hybrid loss (see `docs/phase4/distillation.html`), plus a real infrastructure investigation on this machine's own hardware — teacher setup (Qwen 3.6 35B-A3B MoE via a local Docker/llama.cpp server), measured throughput (prefill vs. decode, ~14x apart), a genuine tokenizer-vocabulary mismatch between teacher (248,320 tokens) and student (32,000 tokens) that rules out direct soft-label KL distillation on this architecture, and a student size decision (300M, measured to fit cleanly) — all documented in `docs/phase4/sprint-plan.md`.

**Key finding:** the teacher/student tokenizer mismatch means true soft-label distillation isn't viable here without either re-tokenizing to a shared (248K-token) vocabulary — which doesn't fit this hardware at any usable batch size — or falling back to a much weaker hard-label-only signal. Combined with the real generation cost (teacher-forced logit extraction is fast, ~1120 tok/s measured, but still hours-to-a-day+ of sustained GPU/CPU load for a meaningful token budget), running a full production distillation here wasn't worth the compute and electricity for the signal it would likely produce.

A real, task-focused distillation project (e.g. distilling toward a specific application like customer support) is the natural place to actually run this, where the teacher/student vocabulary can be chosen to match from the start and the target task justifies the compute.

**Deliverable:** Understanding of distillation concepts and the real infrastructure constraints of doing it on this hardware — not a trained distilled model.

---

## Phase 5: Evaluation & Feedback Loop — "Close the Circle"

**Objective:** Benchmark, analyze errors, generate targeted data, retrain. The full RLHF loop done manually.

### Task 5.1: Set up lm-eval-harness
- EleutherAI's lm-evaluation-harness
- Benchmarks: HellaSwag, PIQA, Winogrande, ARC-Easy/Challenge, GSM8K (math), HumanEval (code)
- Compare: your 75M, 150M, distilled model, GPT-2 124M, SmolLM 135M

### Task 5.2: Error analysis
- Find 50 worst examples per benchmark
- Categorize: math, commonsense, hallucination, tokenization

### Task 5.3: Targeted synthetic data (feedback loop)
- Per error category, generate 5K-20K targeted examples with teacher
- Math → step-by-step solutions
- Commonsense → "explain the reasoning"
- Hallucination → factuality with citations
- Fine-tune on targeted data

### Task 5.4: Re-evaluate and iterate
- Run benchmarks again — did targeted data fix weaknesses?
- Repeat: find new weaknesses → generate → retrain
- **This is the production RLHF loop, understood from first principles.**

**Deliverable:** W&B dashboard with all model scores, error categories, and improvement trajectory.

---

## Tools & Infrastructure Checklist

```
[✓] GitHub repo: github.com/Ishaanred/slm-from-scratch
[ ] PyTorch 2.5+ with CUDA 12.x
[ ] HuggingFace: transformers, datasets, tokenizers, accelerate
[ ] Weights & Biases (wandb) — free tier
[ ] lm-evaluation-harness (EleutherAI)
[ ] nanoGPT (reference — read it, don't fork it)
[ ] llama.cpp (Qwen 35B MoE 4-bit inference)
[ ] Qwen 3.6 35B MoE (Apache 2.0 teacher)
[ ] DeepSeek Flash v4 API key (backup teacher)
[ ] datatrove or text-dedup (data filtering)
```

---

## Weekly Rhythm

- **Mon-Tue:** Active work — code, data prep, analysis
- **Wed-Fri:** GPU-idle — start training runs, switch to FrameWeaver, check W&B
- **Weekend:** Review results, write findings, plan next week

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| 5070 Ti OOM for 300M+ | Gradient accumulation, activation checkpointing, bf16 |
| Training crashes at 80% | Checkpoint every 1000 steps, resume |
| Teacher inference competes with training GPU | Use DeepSeek API when GPU is training |
| Data pipeline bottleneck | Process data while training, overlap I/O + compute |
| Motivation dips | Every run gives a loss curve. Loss curves = progress. Ship weekly. |

---

## What NOT to Do

- ❌ Don't write tokenizer in pure Python — HF tokenizers is 100× faster
- ❌ Don't try to match GPT-2 124M quality (40B tokens, much more compute)
- ❌ Don't skip data inspection — read 100 samples
- ❌ Don't train fp32 — bf16 is 2× faster, same quality
- ❌ Don't hyperparameter-tune before you have a working pipeline
- ❌ Don't wait for perfect data — train, see what breaks, fix

---

## Portfolio Output

1. Public repo with clean training code
2. Scaling law plots with your data points
3. Distilled model beating from-scratch baseline
4. Benchmark comparisons vs GPT-2, SmolLM
5. Blog-ready 2-page writeup
6. End-to-end understanding of the LLM training pipeline

**Hireable ML engineering knowledge, compressed into 2 months.**
