# Phase 2 Results

## What was built

- A byte-level BPE tokenizer trained on a filtered OpenWebText sample, at three vocab sizes (8K/16K/32K), via `src/phase2/train_tokenizer.py`
- A filtering pipeline (`data/phase2/filter.py`): fastText language-ID (English only, confidence > 0.65), length/quality heuristics (drop <100 words, drop high symbol/digit ratio), exact-dedup by content hash
- The clean corpus re-tokenized with the 32K BPE tokenizer into `train.bin`/`val.bin`
- `src/phase2/train.py` — same architecture as Phase 1, wired to the 32K vocab, to verify the new pipeline actually trains

**Note on process:** the tokenizer training script, filter script, and evaluator were written in a prior session with AI assistance rather than hand-written — see `CLAUDE.md`'s learning boundary, which calls out tokenizer training specifically as something to run and interpret yourself. The concepts (BPE merging, compression tradeoffs, filter heuristics) were reviewed and understood after the fact, but the code itself wasn't personally authored this round.

---

## Tokenizer compression comparison

Measured on held-out OpenWebText documents (not in the training sample), tokens per 1,000 characters — fewer is better:

```
GPT-2  (50,257)  227.2 tok / 1k chars   (baseline)
yours  ( 8,000)  267.4 tok / 1k chars   (-17.7% vs GPT-2, worse)
yours  (16,000)  244.5 tok / 1k chars   (-7.6% vs GPT-2, worse)
yours  (32,000)  230.0 tok / 1k chars   (-1.2% vs GPT-2, worse)
```

**Finding:** the 32K tokenizer doesn't beat GPT-2's compression on general web text — it's marginally worse. Likely cause: GPT-2's tokenizer was trained on a much larger, more diverse corpus, so it generalizes better to held-out text than a tokenizer trained on the smaller, filtered `clean.txt` sample used here. Bigger/more diverse training corpus for the tokenizer itself would probably close this gap.

---

## Training verification run

**Config:** 8 layers, 8 heads, 512 embedding dim (58.5M params), 32K vocab
**Steps:** 5,000
**Val loss:** 5.23 (final, at iter 4500)
**Time:** ~15 min on RTX 5070 Ti, ~150K tok/s

Raw val loss (5.23) looks almost identical to Phase 1's 77M run (5.24) — that's a coincidence, not evidence the pipelines are equivalent, since loss is per-token and the two runs use different-sized tokens. Converting to bits-per-byte (loss × tokens-per-byte / ln 2) using the compression numbers above:

```
Phase 1 (GPT-2 tokenizer, 227.2 tok/1k chars):  1.718 bits/byte
Phase 2 (32K tokenizer, 230.0 tok/1k chars):    1.736 bits/byte
```

**This comparison is confounded and should not be read as "the tokenizer made the model worse."** Phase 1 trained on the full OpenWebText corpus (~8.5B tokens). Phase 2's `train.bin` was built from `data/phase2/clean.txt`, which comes from the 200K-document / 1-2GB sample `sample_corpus.py` pulls — a corpus explicitly scoped for training the *tokenizer*, not the language model (see the comment in `data/phase2/prepare.py`, which notes the full-corpus version of that script exists in git history but isn't what ran). After tokenizing, that sample yields only ~214M tokens — **about 40x smaller** than Phase 1's corpus.

So the two training runs differ in two ways at once: tokenizer compression (a real but small ~1.2% effect, isolated cleanly above) and training-corpus size/diversity (a ~40x difference, not controlled for at all). A 0.018 bits/byte gap is easily explained by the corpus-size difference alone — this result does not support "the custom tokenizer hurt training." It only supports the standalone tokenizer-compression finding above.

**What a clean comparison would require:** re-tokenize the *full* OpenWebText corpus with the 32K tokenizer (the full-corpus version of `prepare.py` mentioned in its own comment) and train on that, so both runs see comparable data volume. That's the actual next step if this comparison matters going forward — right now the honest conclusion is "inconclusive on training impact, tokenizer compression is slightly worse in isolation."

---

## Next

Phase 3 — scaling law experiments across model sizes and token counts. **Deferred, not skipped:** the clean Phase 1 vs Phase 2 comparison (re-tokenizing the full OpenWebText corpus with the 32K tokenizer, then training at matched token counts) isn't worth doing as a standalone step now — Phase 3's experiment grid already runs controlled comparisons across token counts, so re-tokenizing the full corpus becomes step 1 of that phase's data prep instead of duplicated work here.
