# Transformers

[← back to index](README.md)

## The idea, in one analogy

Imagine reading a sentence one word at a time, and before guessing the next word, you're allowed to glance back at every word you've already read and decide how much attention each one deserves. Take "The trophy didn't fit in the suitcase because *it* was too big": to figure out what "it" refers to, you weigh "trophy" and "suitcase" very differently depending on the rest of the sentence. That weighing act is "attention," and a transformer is a stack of layers that do this weighing, over and over, building up a richer understanding of the sequence at each layer.

## What's actually inside one

A GPT-style transformer (decoder-only, the kind used for text generation) is built from four pieces, stacked:

**Token + position embeddings.** Each token (see [tokenization.md](tokenization.md)) gets converted to a vector of numbers. Since attention has no built-in sense of order, a separate "position" vector is added so the model knows token 3 came before token 7.

**Causal self-attention.** Each position produces a query ("what am I looking for"), and every position produces a key ("what do I contain") and a value ("what do I actually offer"). Attention compares each query against every earlier key, turns the result into weights, and mixes the values accordingly. "Causal" means a position can only attend to itself and earlier positions; it's not allowed to see the future, since the whole point is predicting what comes next.

**MLP (feed-forward block).** After attention mixes information across positions, an MLP processes each position independently: typically expand to 4x the width, apply a nonlinearity, project back down. This is where a lot of the model's "knowledge" is thought to live.

**Residual connections + normalization.** Attention and MLP outputs are added back to their input (not replacing it) at every step, and normalized, which is what makes it possible to stack dozens of these layers without training collapsing.

A "block" is one attention + one MLP (with their residual connections). A GPT model is N of these blocks stacked, followed by a final projection back to vocabulary size, turning the model's internal representation into a probability distribution over "what token comes next."

<details><summary>∑ the actual math</summary>

Attention for a single head:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

Q, K, V are learned linear projections of the input. The softmax turns raw similarity scores into a probability distribution over positions to attend to; dividing by √d_k keeps the scores from growing too large as dimensionality increases. Multi-head attention runs several of these in parallel with different learned projections, then concatenates the results, letting different heads specialize (one might track syntax, another long-range coreference, etc).

Causal masking sets the attention score to −∞ for any (query position, key position) pair where the key position is in the future, before the softmax, so those positions get exactly zero weight.
</details>

## Why it beats what came before

Before transformers, sequence models (RNNs, LSTMs) processed tokens one at a time, in order, which meant training couldn't parallelize across the sequence length and long-range dependencies had to survive being passed through many sequential steps (they often didn't). Attention lets every position see every other position directly, in one operation, and lets the whole sequence be processed in parallel during training.

## Where to look further

- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762): the original paper.
- Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT): the clearest minimal implementation to read alongside this.
- This project's own from-scratch build: [`slm-from-scratch` Phase 1](../docs/phase1/how-transformers-work.html) and [`src/phase1/model.py`](../src/phase1/model.py).
