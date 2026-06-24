"""
src/train.py — Training loop scaffolding. Core train_step() is yours to write.

This file sets up everything around training:
  - Config parsing
  - DataLoader
  - Optimizer + LR scheduler
  - W&B logging
  - Checkpoint save/resume
  - The outer training loop

What you fill in:
  - train_step(): forward pass, loss computation, backward, gradient accumulation
"""

import os
import math
import time
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TrainConfig:
    # Data
    data_dir: str = "data/phase1/"
    block_size: int = 1024

    # Model
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 8  # effective batch = batch_size * grad_accum
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    log_interval: int = 10

    # Optimizer
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule (cosine with warmup)
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 6e-5

    # Checkpointing
    out_dir: str = "checkpoints/phase1/"
    resume_from: str = ""

    # Logging
    wandb_project: str = "slm-from-scratch"
    wandb_run_name: str = "gpt-75m"


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
# train_step — YOU WRITE THIS
# ---------------------------------------------------------------------------

def train_step(model, optimizer, x, y, config, scaler):
    """
    One optimizer step using gradient accumulation.

    Args:
        model: GPT instance
        optimizer: AdamW
        x, y: input and target token tensors, shape [batch, seq_len]
        config: TrainConfig
        scaler: torch.cuda.amp.GradScaler for bf16 training

    Returns:
        loss (float): the loss value for logging

    Hints:
        - Use torch.autocast("cuda", dtype=torch.bfloat16)
        - Accumulate gradients over config.gradient_accumulation_steps
        - Clip gradients with torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        - Step optimizer, zero_grad
    """
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
# Main training loop — provided (calls your train_step)
# ---------------------------------------------------------------------------

def main():
    config = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config), resume="allow")

    # Model
    model_config = GPTConfig(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
    )
    model = GPT(model_config).to(device)
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

        # Train step (your code)
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
