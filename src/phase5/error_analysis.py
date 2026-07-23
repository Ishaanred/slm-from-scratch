"""
Pulls wrong-answer examples for manual error analysis (Phase 5, Step 3).

This only extracts and formats — categorizing *why* each example was
wrong (tokenization artifact, reasoning gap, scale limitation) is the
analysis step that's yours to do, per docs/phase5/sprint-plan.md.

Run: python src/phase5/error_analysis.py [task] [checkpoint_name] [n]
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_DIR = os.path.join(ROOT, "docs", "phase5")


def load_samples(checkpoint_name, task):
    path = os.path.join(DOCS_DIR, f"eval_samples_{checkpoint_name}.json")
    with open(path) as f:
        return json.load(f)[task]


def format_example(sample, task):
    doc = sample["doc"]
    resps = sample["filtered_resps"]  # list of [logprob, is_greedy] per choice
    logprobs = [r[0] for r in resps]
    picked = logprobs.index(max(logprobs))
    target = sample["target"]

    if task == "arc_easy":
        question = doc["question"]
        choices = doc["choices"]["text"]
        labels = doc["choices"]["label"]
    elif task == "piqa":
        question = doc["goal"]
        choices = [doc["sol1"], doc["sol2"]]
        labels = ["1", "2"]
    elif task == "hellaswag":
        question = doc.get("ctx", doc.get("query", ""))
        choices = doc["endings"]
        labels = [str(i) for i in range(len(choices))]
    else:
        question = str(doc)
        choices = []
        labels = []

    return {
        "question": question,
        "choices": list(zip(labels, choices)),
        "correct_label": labels[target] if target < len(labels) else target,
        "model_picked_label": labels[picked] if picked < len(labels) else picked,
        "logprobs": logprobs,
    }


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "arc_easy"
    checkpoint_name = sys.argv[2] if len(sys.argv) > 2 else "phase3_150m_500m"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    samples = load_samples(checkpoint_name, task)
    # use raw "acc" (unnormalized), matching the acc,none numbers already
    # reported in docs/phase5/eval_results.json and evaluation.html
    wrong = [s for s in samples if s.get("acc", 0) == 0.0]

    print(f"=== {checkpoint_name} / {task}: {len(wrong)} wrong of {len(samples)} ({len(wrong)/len(samples)*100:.1f}%) ===\n")

    for s in wrong[:n]:
        ex = format_example(s, task)
        print(f"Q: {ex['question']}")
        for label, text in ex["choices"]:
            marker = ""
            if label == ex["correct_label"]:
                marker += " [CORRECT]"
            if label == ex["model_picked_label"]:
                marker += " [MODEL PICKED]"
            print(f"  ({label}) {text}{marker}")
        print()


if __name__ == "__main__":
    main()
