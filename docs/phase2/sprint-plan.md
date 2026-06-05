# Phase 2 Sprint Plan — Tokenizer + Data Engineering

**Goal:** Replace the GPT-2 tokenizer with one you trained, and build a data pipeline that produces cleaner training data than raw OpenWebText.

**Why this matters more than more training steps:** A better tokenizer compresses text more efficiently (fewer tokens per idea), and cleaner data means the model learns from signal instead of noise. These improvements carry through every future training run.

---

## Sprint 2A — Train a BPE Tokenizer

**What you're building:** A byte-pair encoding tokenizer trained on your own corpus, using HuggingFace `tokenizers`.

### Tasks

**2A.1 — Pick a training corpus**
- Use a sample of the existing OpenWebText data (1-2GB is enough for a tokenizer)
- Or download FineWeb-Edu sample (higher quality text, better for tokenizer vocab)

**2A.2 — Train the tokenizer**
- File: `src/tokenizer_/train_tokenizer.py`
- Use `tokenizers.ByteLevelBPETokenizer` or `tokenizers.trainers.BpeTrainer`
- Vocab sizes to compare: 8K, 16K, 32K
- Add special tokens: `<|endoftext|>`, `<|pad|>`

**2A.3 — Evaluate and compare**
- Compression ratio: tokens per 1000 characters vs GPT-2 tokenizer
- Tokenize the same sentence with both — see what changes
- Pick the vocab size that gives the best compression without being too large

**2A.4 — Re-tokenize the dataset**
- Update `data/prepare.py` to use your tokenizer instead of GPT-2
- Regenerate `train.bin` and `val.bin`

**Deliverable:** A saved tokenizer in `src/tokenizer_/` + updated data pipeline using it.

---

## Sprint 2B — Data Filtering Pipeline

**What you're building:** A pipeline that takes raw web text and produces cleaner training data.

### Tasks

**2B.1 — Quality filtering**
- Remove documents under 100 words
- Remove documents with high symbol/number ratio (spam, SEO junk)
- Tool: simple heuristics with `datasets` + `multiprocessing`

**2B.2 — Deduplication**
- Exact dedup: hash each document, remove duplicates
- Near-dedup: MinHash LSH (use `text-dedup` library)
- OpenWebText already has some dedup but not enough

**2B.3 — Language filtering**
- Keep only English documents
- Tool: `fasttext` language detection model (fast, runs on CPU)

**2B.4 — Inspect the data**
- Open 100 random samples before and after filtering
- See what got removed — this is where you learn what's actually in the dataset

**Deliverable:** A filtering script `data/filter.py` that produces a cleaner dataset.

---

## Sprint 2C — Synthetic Data (Stretch Goal)

Only tackle this after 2A and 2B are done. Synthetic data generation is optional for Phase 2 — it's the focus of Phase 4.

**2C.1 — Set up Qwen 3.6 35B MoE locally via llama.cpp**
- Download model (4-bit quantized, fits on 16GB VRAM)
- Generate 10K text continuations from high-quality seed text
- Add to dataset mix at 10-15%

---

## What to skip for now

- PII removal (presidio) — useful in production, overkill for a learning project
- Perplexity-based filtering — needs a trained reference model, do in Phase 3
- Full synthetic data pipeline — that's Phase 4

---

## Files to create

```
src/
└── tokenizer_/
    ├── train_tokenizer.py    # Train BPE tokenizer
    └── evaluate_tokenizer.py # Compare compression ratios
data/
└── filter.py                 # Quality + dedup + language filtering
```

---

## Order of work

1. Train tokenizer (2A) — can do in one session, ~2-3 hours
2. Data filtering (2B) — can run overnight while doing other things
3. Re-tokenize with new tokenizer (2A.4) — after filtering is done
4. Short training run to verify the new pipeline works — 1K steps, check loss is similar to Phase 1
