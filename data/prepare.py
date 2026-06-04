"""
Tokenize OpenWebText into train.bin and val.bin.

Output: two memory-mapped uint16 files the training loop reads directly.
Run once: python data/prepare.py
"""

import os
import numpy as np
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
from tqdm import tqdm

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DATA_DIR, "openwebtext")
OUT_DIR = os.path.join(DATA_DIR)

VAL_SIZE = 5000  # documents held out for validation

def tokenize(example, tokenizer):
    ids = tokenizer.encode(example["text"])
    ids.append(tokenizer.eos_token_id)
    return {"ids": ids, "len": len(ids)}

def write_bin(token_lists, path):
    total = sum(len(ids) for ids in token_lists)
    arr = np.empty(total, dtype=np.uint16)
    idx = 0
    for ids in token_lists:
        arr[idx : idx + len(ids)] = ids
        idx += len(ids)
    arr.tofile(path)
    print(f"Wrote {total:,} tokens → {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")

def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    print("Loading dataset...")
    ds = load_from_disk(DATASET_PATH)
    print(f"{len(ds):,} documents")

    print("Tokenizing (this takes ~20-30 min)...")
    ds = ds.map(
        lambda ex: tokenize(ex, tokenizer),
        remove_columns=["text"],
        num_proc=8,
        desc="tokenizing",
    )

    val_ds = ds.select(range(VAL_SIZE))
    train_ds = ds.select(range(VAL_SIZE, len(ds)))

    print("Writing train.bin...")
    write_bin(train_ds["ids"], os.path.join(OUT_DIR, "train.bin"))

    print("Writing val.bin...")
    write_bin(val_ds["ids"], os.path.join(OUT_DIR, "val.bin"))

    print("Done.")

if __name__ == "__main__":
    main()
