# SLM From Scratch

> Train small language models (75M → 500M params) from scratch on a single RTX 5070 Ti (16GB VRAM).

**Learning goals:** Data engineering, tokenization, scaling laws, knowledge distillation, evaluation benchmarks, and closed-loop improvement — every step of the LLM training pipeline, end to end.

## Hardware

- **GPU:** NVIDIA RTX 5070 Ti (16GB VRAM)
- **Teacher models (local):** Qwen 3.6 35B MoE (runs on 5070 Ti), Qwen 2.5 7B
- **Teacher models (API fallback):** DeepSeek Flash v4, Groq (Qwen 2.5 7B)
- **Cloud (if needed):** RunPod 5090 for heavy distillation runs (~10 hours)

## Roadmap

| Phase | Focus | Timeline |
|-------|-------|----------|
| **1. NanoGPT** | Build a transformer from scratch, train on OpenWebText | Week 1 |
| **2. Data Pipeline** | BPE tokenizer, dedup/filter, synthetic data generation | Week 2-3 |
| **3. Scaling Laws** | 10 controlled experiments across model sizes and token counts | Week 3-5 |
| **4. Distillation** | Distill Qwen 35B MoE → 150M/300M student | Week 5-7 |
| **5. Evaluation Loop** | Benchmark, error analysis, targeted retraining | Week 7-8 |

Full plan: [`docs/plan.md`](docs/plan.md)

## Stack

- **Training:** PyTorch 2.5+, HuggingFace (transformers, datasets, tokenizers), nanoGPT (reference)
- **Teacher inference:** llama.cpp (Qwen 35B MoE 4-bit)
- **Tracking:** Weights & Biases
- **Evaluation:** lm-evaluation-harness (EleutherAI)
- **Data:** datatrove / text-dedup, C4, FineWeb-Edu, OpenWebText

## Quick Start

```bash
# Environment
pip install torch transformers datasets tokenizers wandb tqdm einops

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 5070 Ti

# Phase 1: Train nanoGPT clone
python train.py --config configs/75m.yaml
```

## Repository Structure (planned)

```
slm-from-scratch/
├── configs/           # YAML configs for model sizes and training runs
├── src/
│   ├── model.py       # GPT model (nanoGPT-style)
│   ├── trainer.py     # Training loop
│   ├── tokenizer_/    # Custom tokenizer training
│   ├── data/          # Data pipeline, filtering, synthetic generation
│   └── eval/          # Evaluation harness wrappers
├── experiments/       # Scaling law experiment logs and plots
├── docs/
│   └── plan.md        # Full 2-month roadmap
└── notebooks/         # Analysis notebooks
```
