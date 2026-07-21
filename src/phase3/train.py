"""
src/phase3/train.py — same harness as src/phase2/train.py, parameterized via
CLI args so the same file runs every Phase 3 config instead of one hardcoded
size per copy-pasted file. train_step() is untouched from src/phase2/train.py
— that part is not this file's to change.

Usage:
  python src/phase3/train.py --n_layer 8 --n_head 8 --n_embd 576 \
      --max_iters 61035 --wandb_run_name phase3-75m-500m --out_dir checkpoints/phase3/75m_500m/
"""

import os
import math
import time
import argparse
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TrainConfig:
    # Data
    data_dir: str = "data/phase2/full/"
    block_size: int = 1024

    # Model
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 576
    vocab_size: int = 32_000   # MUST match the tokenizer (Sprint 2A: 32K BPE)
    dropout: float = 0.0

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 8  # effective batch = batch_size * grad_accum
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    log_interval: int = 50

    # Optimizer
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule (cosine with warmup) — lr_decay_iters tracks max_iters
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 6e-5

    # Checkpointing
    out_dir: str = "checkpoints/phase2/"
    resume_from: str = ""

    # Logging
    wandb_project: str = "slm-from-scratch"
    wandb_run_name: str = "gpt-phase2-32k"
    no_compile: bool = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_layer", type=int, default=TrainConfig.n_layer)
    p.add_argument("--n_head", type=int, default=TrainConfig.n_head)
    p.add_argument("--n_embd", type=int, default=TrainConfig.n_embd)
    p.add_argument("--max_iters", type=int, default=TrainConfig.max_iters)
    p.add_argument("--data_dir", type=str, default=TrainConfig.data_dir)
    p.add_argument("--out_dir", type=str, default=TrainConfig.out_dir)
    p.add_argument("--wandb_run_name", type=str, default=TrainConfig.wandb_run_name)
    p.add_argument("--resume_from", type=str, default=TrainConfig.resume_from)
    p.add_argument("--no_compile", action="store_true", help="skip torch.compile() (crash-isolation test)")
    args = p.parse_args()

    config = TrainConfig()
    config.n_layer = args.n_layer
    config.n_head = args.n_head
    config.n_embd = args.n_embd
    config.max_iters = args.max_iters
    config.lr_decay_iters = args.max_iters
    config.no_compile = args.no_compile
    config.data_dir = args.data_dir
    config.out_dir = args.out_dir
    config.wandb_run_name = args.wandb_run_name
    config.resume_from = args.resume_from
    return config


# ---------------------------------------------------------------------------
# LR schedule (cosine decay with linear warmup) — provided
# ---------------------------------------------------------------------------

def get_lr(iter: int, config: TrainConfig) -> float:
    if iter < config.warmup_iters:
        return config.learning_rate * iter / config.warmup_iters
    if iter > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (iter - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """Memory-mapped binary token file. Each sample is a random block_size window."""

    def __init__(self, data_path: str, block_size: int):
        import numpy as np
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        # cap at 100k so the DataLoader never tries to shuffle billions of indices
        return min(100_000, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        # pick a random position each time so we see the full dataset over training
        import numpy as np
        i = np.random.randint(0, len(self.data) - self.block_size)
        chunk = torch.from_numpy(self.data[i : i + self.block_size + 1].astype(int))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# ---------------------------------------------------------------------------
# Checkpoint helpers — provided
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, iter_num, val_loss, config, path):
    os.makedirs(config.out_dir, exist_ok=True)
    torch.save({
        "iter_num": iter_num,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iter_num"], ckpt["val_loss"]


# ---------------------------------------------------------------------------
# Evaluation — provided
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, val_loader, eval_iters, device):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# train_step — unchanged from src/phase2/train.py
# ---------------------------------------------------------------------------

def train_step(model, optimizer, x, y, config, scaler):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss = loss / config.gradient_accumulation_steps
    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    return loss.item() * config.gradient_accumulation_steps


# ---------------------------------------------------------------------------
# Main training loop — provided (calls train_step)
# ---------------------------------------------------------------------------

def main():
    config = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Config: n_layer={config.n_layer} n_head={config.n_head} n_embd={config.n_embd} "
          f"max_iters={config.max_iters} data_dir={config.data_dir} out_dir={config.out_dir}")

    wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config), resume="allow")

    # Model
    model_config = GPTConfig(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        vocab_size=config.vocab_size,
    )
    model = GPT(model_config).to(device)
    if not config.no_compile:
        model = torch.compile(model)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {param_count:.1f}M parameters")

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device,
    )
    scaler = torch.amp.GradScaler('cuda')

    # Data
    train_dataset = TokenDataset(os.path.join(config.data_dir, "train.bin"), config.block_size)
    val_dataset = TokenDataset(os.path.join(config.data_dir, "val.bin"), config.block_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Resume
    iter_num = 0
    best_val_loss = float("inf")
    if config.resume_from:
        iter_num, best_val_loss = load_checkpoint(config.resume_from, model, optimizer, device)
        print(f"Resumed from iter {iter_num}, val_loss {best_val_loss:.4f}")

    # Training loop
    train_iter = iter(train_loader)
    t0 = time.time()

    for iter_num in range(iter_num, config.max_iters):
        # LR update
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # Eval
        if iter_num % config.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, config.eval_iters, device)
            print(f"iter {iter_num}: val loss {val_loss:.4f}")
            wandb.log({"val/loss": val_loss}, step=iter_num)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iter_num, val_loss, config,
                                 os.path.join(config.out_dir, "best.pt"))

        # Train step
        loss = train_step(model, optimizer, x, y, config, scaler)

        # Log
        if iter_num % config.log_interval == 0:
            dt = time.time() - t0
            tokens_per_sec = config.batch_size * config.block_size * config.log_interval / dt
            print(f"iter {iter_num}: loss {loss:.4f}, lr {lr:.2e}, {tokens_per_sec:.0f} tok/s")
            wandb.log({"train/loss": loss, "train/lr": lr, "perf/tokens_per_sec": tokens_per_sec}, step=iter_num)
            t0 = time.time()

    save_checkpoint(model, optimizer, config.max_iters, best_val_loss, config,
                     os.path.join(config.out_dir, "final.pt"))
    print("Done.")


if __name__ == "__main__":
    main()
