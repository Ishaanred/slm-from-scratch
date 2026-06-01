# SLM From Scratch

> Train small language models (75M → 500M params) from scratch on a single RTX 5070 Ti (16GB VRAM).

A hands-on, end-to-end curriculum covering every step of the LLM training pipeline: transformer architecture, tokenization, data engineering, scaling laws, knowledge distillation, and evaluation.

**Hardware:** RTX 5070 Ti (16GB VRAM) · **Timeline:** ~2 months, 4-6 hrs/day

---

## Current Status

| Phase | Status |
|-------|--------|
| 1. NanoGPT — build a transformer from scratch | **← Start here** |
| 2. Tokenizer + Data Engineering | Not started |
| 3. Scaling Law Experiments | Not started |
| 4. Knowledge Distillation | Not started |
| 5. Evaluation & Feedback Loop | Not started |

---

## Phases

### Phase 1 — NanoGPT (Week 1)
Build a GPT-style transformer in ~300 lines of PyTorch and train it on OpenWebText. No HuggingFace abstractions — every line of attention, MLP, and positional embedding is yours.

**You'll understand:** how a transformer actually works, what `scaled_dot_product_attention` does, how loss drives learning.

**Deliverable:** A 75M model generating coherent-ish English after 5K steps.

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

## Stack

| Purpose | Tool |
|---------|------|
| Training | PyTorch 2.5+, `torch.compile()`, Flash Attention (SDPA) |
| Models & data | HuggingFace `transformers`, `datasets`, `tokenizers` |
| Teacher inference | llama.cpp (Qwen 3.6 35B MoE, 4-bit) |
| Teacher API fallback | DeepSeek Flash v4, Groq |
| Experiment tracking | Weights & Biases |
| Evaluation | EleutherAI `lm-evaluation-harness` |
| Data filtering | `datatrove` / `text-dedup` |
| Cloud (heavy runs) | RunPod 5090 |

---

## Repository Structure

```
slm-from-scratch/
├── configs/              # YAML configs per model size and training run
├── src/
│   ├── model.py          # GPT model — Phase 1 entry point
│   ├── trainer.py        # Training loop
│   ├── tokenizer_/       # BPE tokenizer training — Phase 2
│   ├── data/             # Pipeline: filtering, dedup, synthetic gen — Phase 2
│   └── eval/             # lm-eval-harness wrappers — Phase 5
├── experiments/          # Scaling law logs and plots — Phase 3
├── notebooks/            # Analysis and visualization
└── docs/
    └── plan.md           # Full 2-month roadmap with task breakdowns
```

---

## Setup

```bash
pip install torch transformers datasets tokenizers wandb tqdm einops

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 5070 Ti
```

Full task breakdown: [`docs/plan.md`](docs/plan.md)
