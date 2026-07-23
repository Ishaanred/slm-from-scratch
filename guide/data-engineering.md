# Data engineering

[← back to index](README.md)

## The idea, in one analogy

If you learned to write by reading a library where every tenth book was a spam email and half the books had entire chapters duplicated, you'd waste a huge amount of your reading budget on noise, and you'd overlearn whatever got repeated most. A language model has exactly the same problem: raw scraped text is full of junk, duplication, and boilerplate, and every token spent on it is a token not spent learning something useful. Data engineering is the filtering pass that fixes this before training ever starts.

## The usual pipeline

**Language identification.** Web scrapes are multilingual by default; if you're training a single-language model, everything else gets filtered out first (a fastText classifier is the common tool for this).

**Quality heuristics.** Cheap, rule-based filters: documents that are mostly boilerplate, too short, too repetitive, or fail basic "does this look like real prose" checks get dropped. Not perfect, but cheap and catches a lot of the worst material.

**Deduplication.** Exact-hash dedup removes byte-identical documents (surprisingly common in web scrapes: the same article gets mirrored across many domains). Near-duplicate detection (commonly MinHash + locality-sensitive hashing) catches documents that are almost identical but not byte-for-byte, which exact hashing misses entirely.

**Re-tokenization.** If you're training your own tokenizer (see [tokenization.md](tokenization.md)), the filtered corpus gets tokenized with it as the last step before training. Filtering before tokenizing means the tokenizer itself is trained on cleaner data too.

## The uncomfortable truth about this step

It's the least glamorous part of the whole pipeline and the one most likely to get skipped or rushed, and it's also frequently the highest-leverage lever available: a smaller model trained on well-filtered data routinely beats a larger model trained on raw scrape, because so much of a raw scrape's tokens are actively harmful (duplication makes the model overfit toward repeated content) rather than merely neutral.

It's also easy to get an inflated sense of your own results here without noticing: if you change your data pipeline and your corpus size changes at the same time, you can't tell whether an improvement came from cleaner data or just having more of it. Any honest before/after comparison needs to hold corpus size fixed, or explicitly call out that it wasn't (a confound worth naming, not quietly absorbing into "the new pipeline is better").

## Where to look further

- The [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) technical report: a detailed, empirical walkthrough of exactly this pipeline at scale, with ablations showing what each filtering step is actually worth.
- [`datatrove`](https://github.com/huggingface/datatrove): Hugging Face's library for large-scale text filtering/dedup pipelines.
- This project's own filtering pipeline and results: [`slm-from-scratch` Phase 2](../docs/phase2/results.md).
