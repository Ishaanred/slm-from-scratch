# SLM From Scratch

> Train small language models (75M → 500M params) from scratch on a single RTX 5070 Ti (16GB VRAM).

A hands-on, end-to-end curriculum covering every step of the LLM training pipeline: transformer architecture, tokenization, data engineering, scaling laws, knowledge distillation, and evaluation.

**Hardware:** RTX 5070 Ti (16GB VRAM) · **Timeline:** ~2 months, 4-6 hrs/day

---

## Current Status

| Phase | Status |
|-------|--------|
| 1. NanoGPT — build a transformer from scratch | **Complete** |
| 2. Tokenizer + Data Engineering | **Complete** |
| 3. Scaling Law Experiments | **Complete** |
| 4. Knowledge Distillation | **Up next** |
| 5. Evaluation & Feedback Loop | Not started |

**Phase 1 — complete:**
- [x] Understand transformer architecture — see [`docs/how-transformers-work.html`](docs/how-transformers-work.html)
- [x] Write `src/model.py` — GPT, CausalSelfAttention, MLP, Block from scratch in PyTorch
- [x] Download and process OpenWebText dataset (~38GB, 8M documents)
- [x] Tokenize data with GPT-2 tokenizer — produces 17GB `train.bin` (~8.5B tokens)
- [x] Write `train_step()` — bfloat16 autocast, gradient accumulation, grad norm clipping
- [x] First training run — 5K steps, val loss 5.12, ~15 min on RTX 5070 Ti
- [x] Text generation — `python src/generate.py --prompt "..."`
- [x] Published the 77M checkpoint to Hugging Face: [redredredredredred/slm-from-scratch-77m](https://huggingface.co/redredredredredred/slm-from-scratch-77m)

**Phase 2 — complete:**
- [x] Train BPE tokenizers at 8K/16K/32K — see [`docs/phase2/tokenization.html`](docs/phase2/tokenization.html)
- [x] Filtering pipeline: fastText language-ID, quality heuristics, exact dedup
- [x] Re-tokenize the clean corpus with the 32K tokenizer
- [x] Verification run — 58.5M params, 5K steps, val loss 5.23
- [x] Results + honest tokenizer-compression comparison — see [`docs/phase2/results.md`](docs/phase2/results.md)

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
Train your own BPE tokenizer. Build a data pipeline that filters by quality and exact-dedups the corpus.

**You'll understand:** why "garbage in, garbage out" is the dominant failure mode in LLM training.

**Deliverable:** A trained tokenizer (32K BPE) + a clean, filtered dataset, verified end-to-end with a training run. See [`docs/phase2/results.md`](docs/phase2/results.md) for the actual numbers.

**Scoped out, reviewed 2026-07-13:** PII removal and synthetic-data generation (Qwen 35B MoE teacher) aren't specific to this phase's learning goal — PII removal is a compliance concern, and synthetic data via a teacher model is Phase 4's actual lesson (distillation). Near-dedup (MinHash LSH) was also skipped; only exact-hash dedup ran. None of this blocks Phase 3.

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

Each phase is self-contained under its own folder (`src/phase1/`, `data/phase1/`).

```bash
# 1. Install dependencies
pip install torch transformers datasets tokenizers wandb tqdm einops

# 2. Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 3. Tokenize OpenWebText with the GPT-2 tokenizer -> data/phase1/{train,val}.bin (~20-30 min)
python data/phase1/prepare.py

# 4. Train (writes to checkpoints/phase1/), ~15 min for 5K steps on RTX 5070 Ti
python src/phase1/train.py

# 5. Generate text from the trained checkpoint
python src/phase1/generate.py --prompt "The meaning of life is"
```

To skip wandb, run with `WANDB_MODE=disabled` or comment out `wandb.init(...)` in `src/phase1/train.py`.

---

## Running Phase 2

Phase 2 trains its own BPE tokenizer and a cleaner data pipeline (`src/phase2/`, `data/phase2/`).

```bash
# 1. Sample a corpus from OpenWebText  -> data/phase2/sample.txt
python data/phase2/sample_corpus.py

# 2. Filter it (needs fasttext + lid.176.bin)  -> data/phase2/clean.txt
pip install fasttext-wheel
curl -L -o data/phase2/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
python data/phase2/filter.py

# 3. Train BPE tokenizers at 8K/16K/32K  -> src/phase2/bpe_*/
python src/phase2/train_tokenizer.py

# 4. Compare compression vs GPT-2, pick a vocab size
python src/phase2/evaluate_tokenizer.py

# 5. Tokenize the clean corpus  -> data/phase2/{train,val}.bin
python data/phase2/prepare.py

# 6. Verify with a short run (set max_iters small first), writes to checkpoints/phase2/
python src/phase2/train.py
```

See [`docs/phase2/tokenization.html`](docs/phase2/tokenization.html) for the full walkthrough and the reasoning behind each step.

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

Each phase is fully self-contained (its own copy of `model.py`) so phases can be read and run independently.

```
slm-from-scratch/
├── src/
│   ├── phase1/                  # Transformers
│   │   ├── model.py             # GPT transformer — written from scratch
│   │   ├── train.py             # Training loop (GPT-2 vocab, 50,257)
│   │   └── generate.py          # Text generation from a checkpoint
│   └── phase2/                  # Tokenization + Data Engineering
│       ├── model.py             # copy of the transformer
│       ├── train.py             # Training loop (own 32K vocab)
│       ├── train_tokenizer.py   # Train BPE at 8K/16K/32K
│       ├── evaluate_tokenizer.py# Compression ratio vs GPT-2
│       └── bpe_*/               # Trained tokenizers (not in git)
├── data/
│   ├── openwebtext/             # Raw dataset (~38GB, not in git)
│   ├── phase1/
│   │   └── prepare.py           # Tokenize OWT with GPT-2 -> train/val.bin
│   └── phase2/
│       ├── sample_corpus.py     # Carve a text sample
│       ├── filter.py            # Language + quality + dedup
│       └── prepare.py           # Tokenize clean corpus with the 32K BPE
├── checkpoints/
│   ├── phase1/                  # Phase 1 model weights (not in git)
│   └── phase2/                  # Phase 2 model weights (not in git)
└── docs/
    ├── plan.md                  # Full 2-month roadmap
    ├── phase1/                  # how-transformers-work.html + results.md
    └── phase2/                  # tokenization.html + sprint-plan.md
```

---

## Docs

- [`docs/phase1/how-transformers-work.html`](docs/phase1/how-transformers-work.html) — visual guide to transformer internals: tokenization, embeddings, attention, the full nanoGPT codebase with interactive diagrams and hover tooltips. Open in a browser.
- [`docs/phase1/results.md`](docs/phase1/results.md) — Phase 1 results: model configs, val loss, and generated text samples showing progression.
- [`docs/plan.md`](docs/plan.md) — full 2-month roadmap with task breakdowns.
- [`docs/phase2/sprint-plan.md`](docs/phase2/sprint-plan.md) — Phase 2 sprint plan: tokenizer training + data filtering.
- [`docs/phase2/tokenization.html`](docs/phase2/tokenization.html) — full Phase 2 explainer: why Phase 2 differs from Phase 1, BPE step by step, vocab size tradeoffs, byte-level tokenization, compression ratio, the HF tokenizers library, data quality, and the filtering pipeline.
- [`docs/phase2/results.md`](docs/phase2/results.md) — Phase 2 results: tokenizer compression comparison vs GPT-2, the verification training run, and the bits-per-byte analysis of why the raw loss numbers looked deceptively close.
- [`docs/phase3/scaling-laws.html`](docs/phase3/scaling-laws.html) — Phase 3 explainer: what a scaling law is, FLOPs, the Chinchilla rule, undertrained vs overtrained, and the 3-run experiment design.
- [`docs/phase3/results.md`](docs/phase3/results.md) — Phase 3 results: loss vs tokens and loss vs FLOPs plots across all 3 runs, the data table, and a real training-instability spike found in Run 2's curve.
- [`docs/phase4/distillation.html`](docs/phase4/distillation.html) — Phase 4 explainer: hard vs soft labels, temperature scaling, KL divergence, the hybrid loss, and the teacher setup on this machine.
- [`docs/phase4/sprint-plan.md`](docs/phase4/sprint-plan.md) — Phase 4 plan: teacher setup (Qwen 3.6 35B-A3B, already downloaded locally), student size, logit-generation pipeline, and the distillation-loss learning boundary. Not started yet.
