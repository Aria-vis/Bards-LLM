"""
Training script for Bard's Language Model.
Features:
- Memory-mapped zero-copy dataset loading
- Linear warmup + Cosine decay learning rate scheduler
- Dynamic train/val loss estimation
- Gradient clipping & AdamW optimizer with segregated weight decay
- Automatic best-checkpoint serialization
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import time
import numpy as np
import torch

from bard.config import BardConfig
from bard.model import BardLM

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
out_dir = "checkpoints"
eval_interval = 200        # Evaluate loss every N iterations
eval_iters = 25            # Number of batches to average for loss estimation
log_interval = 20          # Print log every N iterations

# Hardware & Batching
batch_size = 16            # Batch size for training
block_size = 128           # Context length
device = "cuda" if torch.cuda.is_available() else "cpu"

# Optimizer & Schedule
max_iters = 1000           # Total training steps (~2 minutes on CPU)
learning_rate = 1e-3       # Peak learning rate
min_lr = 1e-4              # Minimum learning rate (10% of peak)
warmup_iters = 100         # Steps to warm up learning rate
lr_decay_iters = 1000      # Steps to decay learning rate over
weight_decay = 1e-1        # Weight decay for 2D parameters
grad_clip = 1.0            # Clip gradient norm at this threshold

# Data paths
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

# 1. Memory-Mapped Data Loader
def get_batch(split: str):
    """Samples a random batch of inputs (x) and corresponding next-token targets (y)."""
    filename = "train.bin" if split == "train" else "val.bin"
    file_path = os.path.join(data_dir, filename)
    data = np.memmap(file_path, dtype=np.uint16, mode="r")
    
    # Generate random starting indices for sequences
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    
    return x.to(device), y.to(device)


# 2. Loss Estimation Function
@torch.no_grad()
def estimate_loss(model: BardLM):
    """Averages loss over several batches to get an accurate, low-noise measurement."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# 3. Learning Rate Scheduler (Warmup + Cosine Decay)
def get_lr(it: int) -> float:
    # 1. Linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2. Past decay schedule -> return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3. Cosine decay down to min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def main():
    print(f"Executing on device: {device}")
    
    # Initialize Model
    cfg = BardConfig(
        block_size=block_size,
        vocab_size=65,        # 65 unique characters in Tiny Shakespeare
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=False
    )
    model = BardLM(cfg).to(device)
    print(f"Total model parameters: {model.get_num_params() / 1e6:.3f}M")

    # Initialize Optimizer with segregated 2D vs 1D weight decay
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type=device
    )

    best_val_loss = float("inf")
    t0 = time.time()

    print("\nStarting training loop...")
    print("=" * 65)

    for iter_num in range(max_iters + 1):
        # Determine learning rate for this step
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Periodic evaluation & checkpointing
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"Step {iter_num:4d} | Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | LR: {lr:.2e}"
            )

            # Save checkpoint if validation loss improved
            if losses["val"] < best_val_loss and iter_num > 0:
                best_val_loss = losses["val"]
                ckpt_path = os.path.join(out_dir, "best_model.pt")
                checkpoint = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                torch.save(checkpoint, ckpt_path)
                print(f"  --> Saved new best checkpoint to {ckpt_path} (Val Loss: {best_val_loss:.4f})")

        if iter_num == max_iters:
            break

        # Fetch training batch & compute gradients
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient norm clipping
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Step logging
        if iter_num % log_interval == 0 and iter_num > 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num:4d} | loss: {loss.item():.4f} | time: {dt * 1000:.1f}ms")

    print("=" * 65)
    print("Training complete!")


if __name__ == "__main__":
    main()