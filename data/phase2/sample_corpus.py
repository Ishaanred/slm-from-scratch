"""
Sprint 2A, Step 1 - carve a plain-text sample out of OpenWebText.

A couple of GB is plenty to train a tokenizer and to test filters, and it
keeps iteration fast. Writes one document per line so the tokenizer trainer
and the filter (data/filter.py) can read it line by line.

Run once: python data/phase2/sample_corpus.py
"""

import os
from datasets import load_from_disk

DATA = os.path.dirname(os.path.abspath(__file__))
ds = load_from_disk(os.path.join(DATA, "..", "openwebtext"))

N_DOCS = 200_000          # ~1-2 GB of text, enough for a tokenizer
out = os.path.join(DATA, "sample.txt")

written = 0
with open(out, "w", encoding="utf-8") as f:
    for i in range(min(N_DOCS, len(ds))):
        text = ds[i]["text"].strip()
        if text:
            f.write(text.replace("\n", " ") + "\n")   # one doc per line
            written += 1

size_gb = os.path.getsize(out) / 1e9
print(f"wrote {written:,} docs to {out}  ({size_gb:.2f} GB)")
