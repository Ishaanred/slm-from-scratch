"""
Sprint 2A, Step 2 - clean the sampled text.

Three passes as a keep/drop chain. A document survives only if it passes all
three:
  1. language     - English only (fastText lid.176)
  2. quality      - length and symbol/digit ratio heuristics
  3. exact dedup  - drop identical documents (content hash)

Near-deduplication (near-identical rewrites) is a separate, heavier pass best
left to a dedicated library (text-dedup with MinHash) over the output.

Setup:
  pip install fasttext-wheel
  curl -L -o data/phase2/lid.176.bin \
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

Run: python data/phase2/filter.py
"""

import os
import hashlib
import fasttext

DATA = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(DATA, "sample.txt")
OUT = os.path.join(DATA, "clean.txt")

lang = fasttext.load_model(os.path.join(DATA, "lid.176.bin"))

SYMBOLS = set("{}[]<>|@#^*_=~")


def is_english(text):
    label, prob = lang.predict(text, k=1)
    return label[0] == "__label__en" and prob[0] > 0.65


def passes_quality(text):
    if len(text.split()) < 100:                       # too short to be useful
        return False
    n = max(len(text), 1)
    if sum(c in SYMBOLS for c in text) / n > 0.10:    # symbol / SEO junk
        return False
    if sum(c.isdigit() for c in text) / n > 0.30:     # number spam
        return False
    return True


seen = set()


def is_new(text):                                     # exact dedup via hash
    h = hashlib.md5(text.encode()).hexdigest()
    if h in seen:
        return False
    seen.add(h)
    return True


def main():
    kept = total = 0
    dropped = {"lang": 0, "quality": 0, "dup": 0}
    with open(IN, encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            t = line.strip()
            if not t:
                continue
            if not is_english(t):
                dropped["lang"] += 1
                continue
            if not passes_quality(t):
                dropped["quality"] += 1
                continue
            if not is_new(t):
                dropped["dup"] += 1
                continue
            fout.write(t + "\n")
            kept += 1

    pct = 100 * kept / max(total, 1)
    print(f"kept {kept:,}/{total:,} ({pct:.1f}%)")
    print(f"  dropped non-english: {dropped['lang']:,}")
    print(f"  dropped low-quality: {dropped['quality']:,}")
    print(f"  dropped duplicates:  {dropped['dup']:,}")
    print(f"-> {OUT}  ({os.path.getsize(OUT) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
