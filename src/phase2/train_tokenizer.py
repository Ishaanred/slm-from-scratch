"""
Sprint 2A, Step 3 - train byte-level BPE tokenizers on the clean corpus.

Trains three vocabulary sizes so they can be compared in Step 4. Each one
saves vocab.json + merges.txt (via save_model) and a single combined
tokenizer.json (via save), which Step 5 loads.

Run: python src/phase2/train_tokenizer.py
"""

import os
from tokenizers import ByteLevelBPETokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS = os.path.join(ROOT, "data", "phase2", "clean.txt")
TOK_DIR = os.path.join(ROOT, "src", "phase2")

VOCAB_SIZES = (8_000, 16_000, 32_000)
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]


def main():
    for vocab_size in VOCAB_SIZES:
        tok = ByteLevelBPETokenizer()
        tok.train(
            files=[CORPUS],
            vocab_size=vocab_size,
            min_frequency=2,                  # a pair must occur 2+ times to merge
            special_tokens=SPECIAL_TOKENS,
        )
        out = os.path.join(TOK_DIR, f"bpe_{vocab_size}")
        os.makedirs(out, exist_ok=True)
        tok.save_model(out)                   # vocab.json + merges.txt
        tok.save(os.path.join(out, "tokenizer.json"))  # full tokenizer, one file
        print(f"trained {vocab_size:,} -> {out}")


if __name__ == "__main__":
    main()
