"""
Runs lm-eval-harness benchmarks across this project's own checkpoints.

Run: python src/phase5/run_eval.py
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_model  # noqa: F401  (registers "slm_gpt")
import lm_eval

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BPE_TOK = os.path.join(ROOT, "src", "phase2", "bpe_32000", "tokenizer.json")

CHECKPOINTS = {
    # Note: Phase 1's train.py used one fixed output path, so only the last
    # run (77M) survives on disk — the 50M checkpoint was overwritten and
    # isn't available to evaluate separately.
    "phase1_77m": dict(
        checkpoint=os.path.join(ROOT, "checkpoints/phase1/final.pt"),
        tokenizer="gpt2", n_layer=8, n_head=8, n_embd=512, vocab_size=50257,
    ),
    "phase3_75m_500m": dict(
        checkpoint=os.path.join(ROOT, "checkpoints/phase3/75m_500m/final.pt"),
        tokenizer="bpe32k", tokenizer_path=BPE_TOK,
        n_layer=8, n_head=8, n_embd=576, vocab_size=32000,
    ),
    "phase3_75m_2b": dict(
        checkpoint=os.path.join(ROOT, "checkpoints/phase3/75m_2b/final.pt"),
        tokenizer="bpe32k", tokenizer_path=BPE_TOK,
        n_layer=8, n_head=8, n_embd=576, vocab_size=32000,
    ),
    "phase3_150m_500m": dict(
        checkpoint=os.path.join(ROOT, "checkpoints/phase3/150m_500m/final.pt"),
        tokenizer="bpe32k", tokenizer_path=BPE_TOK,
        n_layer=12, n_head=12, n_embd=768, vocab_size=32000,
    ),
}

TASKS = ["piqa", "hellaswag", "arc_easy"]

OUT_DIR = os.path.join(ROOT, "docs", "phase5")


def main():
    all_results = {}
    for name, args in CHECKPOINTS.items():
        print(f"=== {name} ===", flush=True)
        t0 = time.time()
        results = lm_eval.simple_evaluate(
            model="slm_gpt",
            model_args={**args, "block_size": 1024},
            tasks=TASKS,
            log_samples=True,
        )
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s")
        all_results[name] = {
            "scores": results["results"],
            "elapsed_s": dt,
        }
        # save per-checkpoint samples for error analysis
        samples_path = os.path.join(OUT_DIR, f"eval_samples_{name}.json")
        with open(samples_path, "w") as f:
            json.dump(results.get("samples", {}), f, default=str)

        with open(os.path.join(OUT_DIR, "eval_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print("\n=== Summary ===")
    for name, r in all_results.items():
        print(name, r["scores"])


if __name__ == "__main__":
    main()
