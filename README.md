# SLM From Scratch

> Train small language models (75M → 500M params) from scratch on a single RTX 5070 Ti (16GB VRAM).

A hands-on, end-to-end curriculum covering every step of the LLM training pipeline: transformer architecture, tokenization, data engineering, scaling laws, knowledge distillation, and evaluation.

**Hardware:** RTX 5070 Ti (16GB VRAM) · **Timeline:** ~2 months, 4-6 hrs/day

---

## Current Status

| Phase | Status |
|-------|--------|
| 1. NanoGPT — build a transformer from scratch | **Complete** |
| 2. Tokenizer + Data Engineering | Not started |
| 3. Scaling Law Experiments | Not started |
| 4. Knowledge Distillation | Not started |
| 5. Evaluation & Feedback Loop | Not started |

**Phase 1 — complete:**
- [x] Understand transformer architecture — see [`docs/how-transformers-work.html`](docs/how-transformers-work.html)
- [x] Write `src/model.py` — GPT, CausalSelfAttention, MLP, Block from scratch in PyTorch
- [x] Download and process OpenWebText dataset (~38GB, 8M documents)
- [x] Tokenize data with GPT-2 tokenizer — produces 17GB `train.bin` (~8.5B tokens)
- [x] Write `train_step()` — bfloat16 autocast, gradient accumulation, grad norm clipping
- [x] First training run — 5K steps, val loss 5.12, ~15 min on RTX 5070 Ti
- [x] Text generation — `python src/generate.py --prompt "..."`

---

## Phases

### Phase 1 — NanoGPT (Week 1)

Build a GPT-style transformer in ~300 lines of PyTorch and train it on OpenWebText. No HuggingFace abstractions — every line of attention, MLP, and positional embedding written by hand.

**What was built:**
- `CausalSelfAttention` — multi-head attention with Flash Attention (SDPA) and causal mask
- `MLP` — two linear layers with GELU, 4x expansion
- `Block` — pre-norm residual block (attention + MLP)
- `GPT` — token + positional embeddings, N blocks, language model head
- Full training loop with cosine LR schedule, gradient accumulation, W&B logging, checkpointing

**Result:** 50M parameter model trained on OpenWebText. Val loss 5.12 after 5K steps (~80M tokens). Generates grammatically plausible English sentences.

---

### Phase 2 — Tokenizer + Data Engineering (Weeks 2-3)
Train your own BPE tokenizer. Build a data pipeline that deduplicates, filters by quality, removes PII, and generates synthetic data using Qwen 35B MoE as a teacher.

**You'll understand:** why "garbage in, garbage out" is the dominant failure mode in LLM training.

**Deliverable:** A trained tokenizer + a clean, mixed dataset ready for experiments.

---

### Phase 3 — Scaling Law Experiments (Weeks 3-5)
Run 10 controlled training runs across model sizes (75M, 150M, 300M) and token counts (100M–2B). Plot loss vs FLOPs and verify Chinchilla's compute-optimal formula on your own data.

**You'll understand:** how model size, data size, and compute interact — and how to extrapolate.

**Deliverable:** Scaling law plots from your own experiments.

---

### Phase 4 — Knowledge Distillation (Weeks 5-7)
Distill Qwen 3.6 35B MoE into a 150M–300M student using hard labels, soft KL-divergence, and a hybrid loss. Compare against a from-scratch baseline trained on the same data.

**You'll understand:** why distillation works, what KL divergence means at scale, and when it's worth it.

**Deliverable:** A distilled 150M model beating the from-scratch baseline on benchmarks.

---

### Phase 5 — Evaluation & Feedback Loop (Weeks 7-8)
Benchmark on HellaSwag, PIQA, ARC, GSM8K, HumanEval. Find the worst-performing examples, categorize errors, generate targeted synthetic data, retrain, and repeat.

**You'll understand:** the full RLHF loop from first principles — without the RLHF.

**Deliverable:** W&B dashboard tracking improvement across iterations.

---

## Running Phase 1

### 1. Install dependencies
```bash
pip install torch transformers datasets tokenizers wandb tqdm einops
```

### 2. Verify GPU
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 3. Tokenize the dataset
Run once to produce `data/train.bin` and `data/val.bin`:
```bash
python data/prepare.py
```
Takes ~20-30 minutes on CPU.

### 4. (Optional) Set up Weights & Biases
```bash
wandb login
```
To skip wandb, comment out the `wandb.init(...)` line in `src/train.py`.

### 5. Train
```bash
python src/train.py
```
Expected time on RTX 5070 Ti: ~15 min for 5K steps.

### 6. Generate text
```bash
python src/generate.py --prompt "The meaning of life is"
```

---

## Stack

| Purpose | Tool |
|---------|------|
| Training | PyTorch 2.6+, `torch.compile()`, Flash Attention (SDPA) |
| Models & data | HuggingFace `transformers`, `datasets`, `tokenizers` |
| Teacher inference | llama.cpp (Qwen 3.6 35B MoE, 4-bit) |
| Teacher API fallback | DeepSeek Flash v4, Groq |
| Experiment tracking | Weights & Biases |
| Evaluation | EleutherAI `lm-evaluation-harness` |
| Data filtering | `datatrove` / `text-dedup` |
| Cloud (heavy runs) | RunPod |

---

## Repository Structure

```
slm-from-scratch/
├── src/
│   ├── model.py          # GPT transformer — written from scratch
│   ├── train.py          # Training loop with W&B, checkpointing, LR schedule
│   ├── generate.py       # Text generation from a trained checkpoint
│   ├── tokenizer_/       # BPE tokenizer training — Phase 2
│   ├── data/             # Filtering, dedup, synthetic gen — Phase 2
│   └── eval/             # lm-eval-harness wrappers — Phase 5
├── data/
│   ├── prepare.py        # Tokenizes OpenWebText into train.bin/val.bin
│   ├── openwebtext/      # Raw dataset (~38GB, not in git)
│   ├── train.bin         # Tokenized training data (~17GB, not in git)
│   └── val.bin           # Tokenized validation data (not in git)
├── checkpoints/          # Saved model weights (not in git)
├── experiments/          # Scaling law logs and plots — Phase 3
└── docs/
    ├── plan.md           # Full 2-month roadmap with task breakdowns
    └── phase1/
        ├── how-transformers-work.html  # Visual explainer — start here
        └── results.md                  # Model configs, val loss, generated output samples
```

---

## Docs

- [`docs/phase1/how-transformers-work.html`](docs/phase1/how-transformers-work.html) — visual guide to transformer internals: tokenization, embeddings, attention, the full nanoGPT codebase with interactive diagrams and hover tooltips. Open in a browser.
- [`docs/phase1/results.md`](docs/phase1/results.md) — Phase 1 results: model configs, val loss, and generated text samples showing progression.
- [`docs/plan.md`](docs/plan.md) — full 2-month roadmap with task breakdowns.
