"""
Plots for Phase 3's 3 training runs: loss vs tokens, loss vs FLOPs,
train-loss convergence.

Reads the per-iter train loss and per-eval val loss lines straight from the
training logs (no hand-copied numbers) and renders PNGs into docs/phase3/.

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
TRAIN_RE = re.compile(r"iter (\d+): loss ([\d.]+),")


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


def load_train_loss(name):
    """Per-iter train loss (logged every 50 iters), unlike the sparser val loss."""
    path = os.path.join(LOG_DIR, f"phase3_{name}.log")
    iters, losses = [], []
    with open(path) as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return iters, losses


def ema(values, alpha=0.1):
    """Exponential moving average — the raw per-50-iter train loss is too
    noisy to read as a trend on its own."""
    smoothed = []
    prev = values[0]
    for v in values:
        prev = alpha * v + (1 - alpha) * prev
        smoothed.append(prev)
    return smoothed


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

    # --- Plot 3: train loss convergence (raw + EMA), per run ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, cfg in RUNS.items():
        iters, losses = load_train_loss(name)
        ax.plot(iters, losses, color=cfg["color"], linestyle=cfg["ls"],
                linewidth=0.6, alpha=0.25)
        ax.plot(iters, ema(losses), color=cfg["color"], linestyle=cfg["ls"],
                linewidth=2, label=cfg["label"])
    ax.set_yscale("log")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Train loss (log scale)")
    ax.set_title("Phase 3: train loss convergence (faint = raw, bold = EMA)")
    ax.legend(frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "train_loss_convergence.png"), dpi=150)
    plt.close(fig)

    print("Wrote loss_vs_tokens.png, loss_vs_flops.png, and train_loss_convergence.png to", OUT_DIR)


if __name__ == "__main__":
    main()
