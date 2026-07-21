"""
Tokenize the full OpenWebText corpus with the 32K BPE tokenizer, for Phase 3.

data/phase2/prepare.py tokenizes the small filtered sample (~214M tokens) —
enough for a tokenizer, not for Phase 3's 500M-2B token runs. This mirrors
data/phase1/prepare.py's pattern (full raw corpus, no filtering) but with
the 32K BPE tokenizer instead of GPT-2's, so Phase 1 and Phase 2's tokenizers
can finally be compared on equal training-data volume.

Run: python data/phase2/prepare_full.py
"""

import os
import numpy as np
from datasets import load_from_disk
from tokenizers import Tokenizer
from tqdm import tqdm

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(DATA_DIR))
DATASET_PATH = os.path.join(DATA_DIR, "..", "openwebtext")
OUT_DIR = os.path.join(DATA_DIR, "full")
TOK_PATH = os.path.join(ROOT, "src", "phase2", "bpe_32000", "tokenizer.json")

VAL_SIZE = 5000

_tok = None


def tokenize(example):
    global _tok
    if _tok is None:
        _tok = Tokenizer.from_file(TOK_PATH)
    ids = _tok.encode(example["text"]).ids
    ids.append(_tok.token_to_id("<|endoftext|>"))
    return {"ids": ids, "len": len(ids)}


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
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading dataset...")
    ds = load_from_disk(DATASET_PATH)
    print(f"{len(ds):,} documents")

    print("Tokenizing full corpus with the 32K BPE tokenizer (this will take a while)...")
    ds = ds.map(tokenize, remove_columns=["text"], num_proc=8, desc="tokenizing")

    val_ds = ds.select(range(VAL_SIZE))
    train_ds = ds.select(range(VAL_SIZE, len(ds)))

    print("Writing train.bin...")
    write_bin(train_ds["ids"], os.path.join(OUT_DIR, "train.bin"))

    print("Writing val.bin...")
    write_bin(val_ds["ids"], os.path.join(OUT_DIR, "val.bin"))

    print("Done.")


if __name__ == "__main__":
    main()
