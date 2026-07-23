# Tokenization

[← back to index](README.md)

## The idea, in one analogy

Imagine you had to communicate using only a fixed deck of a few thousand flashcards, each with a word or word-fragment printed on it. "Unbelievable" might not have its own card, but "un", "believ", and "able" do, so you hold up three cards instead of one. A tokenizer is that deck: a fixed vocabulary of chunks, and a rule for splitting any input text into a sequence of them.

## Why not just use whole words, or single letters?

Single characters: tiny vocabulary, but every sentence becomes a very long sequence of tokens, and the model has to rebuild "word" as a concept from scratch.

Whole words: short sequences, but the vocabulary needs to be enormous to cover every word form in every language (and it still fails the moment it sees a word it's never seen: a genuinely new brand name, a typo, a rare technical term).

Byte-Pair Encoding (BPE), used by nearly every modern LLM tokenizer, is the middle ground: start with individual characters (or bytes), then repeatedly merge the most frequent adjacent pair into a new token, thousands of times, until you hit a target vocabulary size. Common words end up as single tokens ("the" → 1 token); rare or unfamiliar text falls back to smaller pieces, down to individual bytes in the worst case, so nothing is ever "unencodable."

<details><summary>∑ how the merge algorithm works</summary>

1. Start with the training corpus split into individual characters/bytes, each its own token.
2. Count every adjacent pair of tokens across the whole corpus.
3. Merge the single most frequent pair into a new token (e.g. `t` + `h` → `th`).
4. Repeat, re-counting pairs each time (now `th` can pair with other tokens), until the vocabulary reaches the target size (commonly 32K-100K+ for production models).

The result is a fixed list of merge rules, applied greedily left-to-right at inference time: deterministic, no learned parameters beyond the merge list itself.
</details>

## The decision that actually matters: whose tokenizer?

Using a widely-used pretrained tokenizer (GPT-2's, or a modern one like tiktoken's `cl100k_base`) means compatibility with existing tooling and no training step, but it was built on that project's data distribution, not yours. Training your own BPE tokenizer on your own corpus gets you better compression (fewer tokens per unit of text, which directly reduces training and inference cost) on whatever domain your data actually is, at the cost of an extra pipeline step, and losing drop-in compatibility with anything expecting the original vocabulary.

A concrete, measured example: a domain-specific tokenizer can badly fragment vocabulary it wasn't trained on. In practice, a general-web 32K BPE tokenizer split the word "photosynthesis" into 4 pieces; a science-specific tokenizer trained on the right corpus would likely keep it as one or two. That fragmentation isn't just cosmetic: more tokens per concept means more steps for the model to piece the idea back together, and less context fits in a fixed window.

## Where to look further

- Sennrich et al., ["Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909): the original BPE-for-NLP paper.
- Hugging Face's [`tokenizers`](https://github.com/huggingface/tokenizers) library: the standard tool for training your own.
- This project's own tokenizer build and compression comparison: [`slm-from-scratch` Phase 2](../docs/phase2/tokenization.html).
