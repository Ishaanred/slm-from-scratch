"""
Tokenize the cleaned corpus into train.bin and val.bin using the tokenizer
trained in Sprint 2A (Step 5).

Reads data/clean.txt (one document per line, produced by data/filter.py) and
encodes it with the byte-level BPE tokenizer chosen in Step 4. Output: two
memory-mapped uint16 files the training loop reads directly.

To re-tokenize the full raw OpenWebText instead, see the git history for the
earlier version of this file that read data/openwebtext via load_from_disk.

Run: python data/phase2/prepare.py
"""

import os
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(DATA_DIR))
CORPUS = os.path.join(DATA_DIR, "clean.txt")

VOCAB = 32_000                      # the size picked in Step 4
TOK_PATH = os.path.join(ROOT, "src", "phase2", f"bpe_{VOCAB}", "tokenizer.json")

VAL_SIZE = 5000                     # documents held out for validation
BATCH = 1000                        # docs per encode_batch call


def write_bin(token_lists, path):
    total = sum(len(ids) for ids in token_lists)
    arr = np.empty(total, dtype=np.uint16)
    idx = 0
    for ids in token_lists:
        arr[idx : idx + len(ids)] = ids
        idx += len(ids)
    arr.tofile(path)
    print(f"Wrote {total:,} tokens -> {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")


def main():
    tok = Tokenizer.from_file(TOK_PATH)
    eot = tok.token_to_id("<|endoftext|>")
    print(f"tokenizer: {TOK_PATH}  (vocab {tok.get_vocab_size():,}, eot id {eot})")

    docs = [line.strip() for line in open(CORPUS, encoding="utf-8") if line.strip()]
    print(f"{len(docs):,} documents")

    all_ids = []
    for i in tqdm(range(0, len(docs), BATCH), desc="tokenizing"):
        for enc in tok.encode_batch(docs[i : i + BATCH]):
            ids = enc.ids
            ids.append(eot)            # mark the document boundary
            all_ids.append(ids)

    val = all_ids[:VAL_SIZE]
    train = all_ids[VAL_SIZE:]

    print("Writing train.bin...")
    write_bin(train, os.path.join(DATA_DIR, "train.bin"))
    print("Writing val.bin...")
    write_bin(val, os.path.join(DATA_DIR, "val.bin"))
    print("Done.")


if __name__ == "__main__":
    main()
