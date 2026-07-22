"""
Plots for Phase 3's 3 training runs: loss vs tokens, loss vs FLOPs.

Reads the per-eval val loss lines straight from the training logs (no
hand-copied numbers) and renders two PNGs into docs/phase3/.

Run: python docs/phase3/plot_results.py
"""

import re
import os
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TOKENS_PER_ITER = 8 * 1024  # batch_size * block_size

RUNS = {
    "75m_500m": {"label": "75M @ 500M tok", "params": 69.4e6, "color": "#2a78d6", "ls": "-"},
    "75m_2b":   {"label": "75M @ 2B tok",   "params": 69.4e6, "color": "#2a78d6", "ls": "--"},
    "150m_500m": {"label": "150M @ 500M tok", "params": 135.0e6, "color": "#eb6834", "ls": "-"},
}

VAL_RE = re.compile(r"iter (\d+): val loss ([\d.]+)")


def load_run(name):
    path = os.path.join(LOG_DIR, f"phase3_{name}.log")
    iters, losses = [], []
    with open(path) as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return iters, losses


def main():
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#c9c8c3",
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.grid": True,
        "grid.color": "#e5e4df",
        "grid.linewidth": 0.6,
        "font.size": 11,
    })

    data = {name: load_run(name) for name in RUNS}

    # --- Plot 1: loss vs tokens seen ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, cfg in RUNS.items():
        iters, losses = data[name]
        tokens = [i * TOKENS_PER_ITER for i in iters]
        ax.plot(tokens, losses, color=cfg["color"], linestyle=cfg["ls"],
                linewidth=2, label=cfg["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Tokens seen")
    ax.set_ylabel("Validation loss")
    ax.set_title("Phase 3: validation loss vs. training tokens")
    ax.legend(frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_vs_tokens.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: loss vs FLOPs (C = 6ND) ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, cfg in RUNS.items():
        iters, losses = data[name]
        flops = [6 * cfg["params"] * (i * TOKENS_PER_ITER) for i in iters]
        ax.plot(flops, losses, color=cfg["color"], linestyle=cfg["ls"],
                linewidth=2, label=cfg["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Training compute (FLOPs, C = 6ND)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Phase 3: validation loss vs. training compute")
    ax.legend(frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_vs_flops.png"), dpi=150)
    plt.close(fig)

    print("Wrote loss_vs_tokens.png and loss_vs_flops.png to", OUT_DIR)


if __name__ == "__main__":
    main()
