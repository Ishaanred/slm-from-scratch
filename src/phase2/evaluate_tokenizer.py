"""
Sprint 2A, Step 4 - compare compression ratios.

Metric: tokens produced per 1,000 characters on held-out text. Fewer tokens
means each token carries more text, so more fits in the same context window
and training is cheaper. Compared against GPT-2's tokenizer as a baseline.

Held-out text is taken from documents AFTER the 200k we sampled for training,
so the tokenizers are measured on text they were not trained on.

Run: python src/phase2/evaluate_tokenizer.py
"""

import os
from datasets import load_from_disk
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOK_DIR = os.path.join(ROOT, "src", "phase2")
VOCAB_SIZES = (8_000, 16_000, 32_000)

# held-out text: docs 200_000..201_000 (not in the training sample)
ds = load_from_disk(os.path.join(ROOT, "data", "openwebtext"))
sample = " ".join(ds[i]["text"] for i in range(200_000, 201_000))[:1_000_000]
chars = len(sample)
print(f"held-out sample: {chars:,} chars\n")


def ratio(n_tokens):
    return n_tokens / chars * 1000


gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")
base = ratio(len(gpt2.encode(sample)))
print(f"GPT-2  (50,257) {base:6.1f} tok / 1k chars   (baseline)")

for v in VOCAB_SIZES:
    tok = ByteLevelBPETokenizer(
        os.path.join(TOK_DIR, f"bpe_{v}", "vocab.json"),
        os.path.join(TOK_DIR, f"bpe_{v}", "merges.txt"),
    )
    r = ratio(len(tok.encode(sample).ids))
    delta = (base - r) / base * 100
    print(f"yours  ({v:>6,}) {r:6.1f} tok / 1k chars   ({delta:+.1f}% vs GPT-2)")
