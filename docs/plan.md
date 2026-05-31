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

## Phase 1: NanoGPT Warm-Up — "I Understand Transformers Now"

**Objective:** Build and train a minimal GPT from scratch (~300 lines of PyTorch). This demystifies every line of a transformer — attention, MLP blocks, positional embeddings, loss computation.

**Why NanoGPT and not just HF?** Because when you write `F.scaled_dot_product_attention(q, k, v, is_causal=True)` yourself and it produces coherent text, you *own* the concept. HF abstracts this away.

### Task 1.1: Set up the environment
- Install: `torch`, `transformers`, `datasets`, `tokenizers`, `wandb`, `tqdm`, `einops`
- Verify CUDA: `python -c "import torch; print(torch.cuda.get_device_name(0))"` → should show RTX 5070 Ti
- Create project repo: `git init slm-from-scratch`

### Task 1.2: Implement GPT-2 style transformer (nanoGPT)
- Write `model.py` — ~300 lines
- Components: `CausalSelfAttention`, `MLP`, `TransformerBlock`, `GPT` class
- Use `torch.compile()` for free 20-30% speedup
- Test: forward pass with random input (batch=4, seq_len=1024, vocab=50257, n_layer=6, n_head=6, n_embd=384 → 75M params)
- Verify: param count with `sum(p.numel() for p in model.parameters())`

### Task 1.3: Train on OpenWebText sample
- Download OpenWebText sample (1M docs, ~4GB text)
- Pre-tokenize with GPT-2 tokenizer (for speed, don't train your own tokenizer yet)
- Train 75M model for 5K steps on 5070 Ti (takes ~2-4 hours)
- Generate samples every 500 steps — watch gibberish become English
- **This is the dopamine hit.** You'll see `[PAD][PAD][PAD]The quick brown fox...` and realize you built this.

### Task 1.4: Play with generation
- Temperature, top-k, top-p sampling
- Save checkpoints, load them, generate

**Deliverable:** A 75M model that produces coherent-ish English after 5K steps. You've touched every line of a transformer.

---

## Phase 2: Tokenizer + Data Engineering — "Garbage In, Garbage Out"

**Objective:** Train your own tokenizer and build a data pipeline. This is 60% of the work in real ML.

### Task 2.1: Train a BPE tokenizer
- Use HuggingFace `tokenizers` library (Rust-backed, fast)
- Train on 10-20GB of text (C4, FineWeb-Edu, or The Pile sample)
- Vocabulary sizes to try: 8K, 16K, 32K
- Compare compression ratios (tokens per word) vs GPT-2 tokenizer
- Save tokenizer, test encode/decode roundtrip

### Task 2.2: Data filtering pipeline
- Build a pipeline that:
  - Removes non-English text (fastText language detection)
  - Deduplicates (MinHash LSH or exact n-gram dedup)
  - Filters by quality (perplexity score using small reference model, length heuristics)
  - Removes PII (regex + presidio)
- Tools: `datatrove`, `text-dedup`, or roll your own with `datasets` + `multiprocessing`
- **Key learning:** Look at the data. Actually open 100 random samples.

### Task 2.3: Synthetic data generation with teacher model
- **Teacher option 1 (local):** Qwen 3.6 35B MoE — runs 4-bit on your 5070 Ti. Use when you're not training.
- **Teacher option 2 (API):** DeepSeek Flash v4 — fast, cheap, good quality. Use when GPU is busy training.
- **Teacher option 3 (cloud):** RunPod 5090 for heavy batch generation (~10 hrs, borrow when needed)
- Generate:
  - **Instruction-following pairs:** "Write a function that..." → code
  - **Chain-of-thought reasoning:** "Solve this step by step..." → reasoning traces
  - **Text continuations:** Seed with high-quality text, let teacher continue
- Aim for 50K-200K high-quality synthetic samples

### Task 2.4: Dataset mixing
- Combine: filtered web text (80%) + synthetic (15%) + curated (5% — books, wiki, code)
- Shuffle, tokenize, pack sequences for efficient training
- Save as HuggingFace dataset or memory-mapped binary files

**Deliverable:** A tokenizer you trained + a clean, deduplicated, mixed dataset ready for training.

---

## Phase 3: Scaling Law Experiments — "Chinchilla Was Right"

**Objective:** Run controlled experiments to understand model size vs data size vs loss.

### The Experiment Grid

| Model Size | 100M tok | 500M tok | 1B tok | 2B tok |
|-----------|----------|----------|--------|--------|
| **75M** | Run 1 | Run 2 | Run 3 | Run 4 |
| **150M** | Run 5 | Run 6 | Run 7 | Run 8 |
| **300M** | — | Run 9 | Run 10 | — |

**That's 10 training runs.** 75M @ 1B tokens takes ~6-8 hours. 150M @ 1B tokens takes ~12-16 hours. Total GPU time: ~80-120 hours.

### Task 3.1: Build the training harness
- Mixed precision (bfloat16), gradient accumulation, cosine LR with warmup, gradient clipping
- W&B logging: loss, perplexity, LR, tokens/sec
- Checkpoint resume
- `torch.compile()` + Flash Attention via PyTorch SDPA

### Task 3.2: Run the experiments
- Runs 1-4 (75M): ~2 days on-and-off
- Runs 5-8 (150M): ~4 days
- Pro tip: Start a run, switch to FrameWeaver, check W&B from your phone

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

## Phase 4: Knowledge Distillation — "The Student Surpasses the Master"

**Objective:** Distill from Qwen 35B MoE (or DeepSeek) into your 150M-500M student.

### Task 4.1: Teacher setup
- **Primary:** Qwen 3.6 35B MoE via llama.cpp 4-bit on 5070 Ti
- **Batch mode:** Generate logits while you sleep, cache to disk
- **Cloud fallback:** RunPod 5090 for heavy logit generation (10 hrs should cover 200M-500M tokens)

### Task 4.2: Generate teacher logits
- For each training sample, get teacher's full logit distribution (top-K, K=8192)
- Storage: ~50-100GB for 1B tokens. Sample 200M tokens if disk-constrained.
- Save as memory-mapped .npy files for fast loading during training

### Task 4.3: Distillation loss variants
1. **Hard distillation:** CE with teacher's top-1 token
2. **Soft distillation:** KL divergence (T=2-4)
3. **Hybrid:** 0.5 × KL + 0.5 × CE (learn distribution + true next token)
- Compare all three on 75M/100M tokens first

### Task 4.4: Train the distilled student
- 150M student with best distillation method
- Baseline: same model from scratch on same data
- Compare: loss curves, benchmarks, generation quality

### Task 4.5: Progressive distillation (stretch)
- Qwen 35B → 1.5B → 500M → 150M
- Does multi-hop preserve quality? (Most people skip this — you'll actually know.)

**Deliverable:** Distilled 150M model beating from-scratch baseline. Understanding of KL divergence at scale.

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
