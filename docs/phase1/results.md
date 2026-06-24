# Phase 1 Results

## What was built

A GPT-style transformer written from scratch in PyTorch (~300 lines). No HuggingFace model classes — every component hand-coded:

- `CausalSelfAttention` — multi-head attention with Flash Attention (SDPA) and causal mask
- `MLP` — two linear layers with GELU, 4x hidden dimension expansion
- `Block` — pre-LayerNorm residual block
- `GPT` — token + positional embeddings, stack of blocks, language model head

Training infrastructure: bfloat16 autocast, gradient accumulation, cosine LR with warmup, AdamW with weight decay only on 2D params, W&B logging, checkpoint save/resume.

Dataset: OpenWebText (~8.5B tokens after tokenization with GPT-2 BPE).

---

## Run 1 — 50M parameter model

**Config:** 6 layers, 6 heads, 384 embedding dim  
**Steps:** 5,000  
**Tokens seen:** ~80M  
**Val loss:** 5.12  
**Time:** ~15 min on RTX 5070 Ti  

**Sample output** (`--prompt "The meaning of life is"`):

```
The meaning of life is that the only way we have to do that.
A United Nations and the United States have already been the largest international community in Europe.

At the moment of the country is coming out in part by the Washington Post reported that the US
government has become known for more than 20 years in the U.S. and that means to be an independent report.

But the United States has a potential interest on the region, the European Council has been a new
law enforcement agencies such as the EU for years.

The president is running out of the United States because it has been difficult to make it illegal to get.

"Our goal is to ensure the president is being a leader of the U.S. and there are no problem that
is no longer the other way to get him to him in the U.S."

The two-year-old government was not known for the end of a public inquiry.

But a report, Mr
```

Real English words, grammatically plausible sentences — but no coherent meaning across sentences. Expected at this stage: 80M tokens is far below what a 50M model needs to converge (~1B tokens by Chinchilla).

---

## Run 2 — 77M parameter model

**Config:** 8 layers, 8 heads, 512 embedding dim  
**Steps:** 5,000  
**Tokens seen:** ~80M  
**Val loss:** 5.24  
**Time:** ~20 min on RTX 5070 Ti  

**Sample output** (`--prompt "The meaning of life is"`):

```
The meaning of life is the first ever-to-to-mused of self-lab and the first time period of time.
The difference between these systems, and even the potential difference to be an example of these two seasons.

The reason it might be, but the first time that it's very well worth noting that we can build up a
new one as a result of the whole way we're talking about, but what we need to be in a pretty much
more successful approach than the last time, and the current performance, and the next one.

After every single reason were true, we haven't seen enough time doing it just in the game. It was
just the most important that we had to find the first thing, but because we are going to be sure we
are not going to get the game to the new roster.

That was the only way we haven't made the second time off of the season, and we don't think it's
the biggest difference.
```

**Why 77M is worse than 50M here:** more parameters need more data. Both models saw 80M tokens — the 77M is more undertrained relative to its size. By Chinchilla, a 77M model needs ~1.5B tokens to match a properly-trained 50M. Same token count, smaller model wins. This will reverse once we run longer training in Phase 3.

---

## Published model

The 77M checkpoint is on Hugging Face: [redredredredredred/slm-from-scratch-77m](https://huggingface.co/redredredredredred/slm-from-scratch-77m). Weights as fp32 safetensors, with a model card and the `model.py` needed to load it.

## Next

Phase 2 — train a custom BPE tokenizer and build a cleaner data pipeline.
