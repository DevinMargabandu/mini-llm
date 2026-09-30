#!/usr/bin/env python3
"""Measure the real effect of mixed-precision autocast on training step
throughput, on whatever device is available. Mixed precision mainly pays
off when matmuls are large enough to be compute-bound rather than
dominated by kernel-launch/memory overhead, so this sweeps a couple of
model sizes rather than asserting one number holds everywhere.
"""
from __future__ import annotations

import time

import torch

from minillm.config import ModelConfig, TrainConfig
from minillm.model.transformer import MiniLLM
from minillm.train.train import autocast_context, pick_device


def bench(model_cfg: ModelConfig, batch_size: int, mixed_precision: bool, device, steps: int = 30):
    train_cfg = TrainConfig(mixed_precision=mixed_precision, amp_dtype="bf16")
    torch.manual_seed(0)
    model = MiniLLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    autocast_fn = autocast_context(device, train_cfg)

    x = torch.randint(0, model_cfg.vocab_size, (batch_size, model_cfg.context_length), device=device)
    y = torch.randint(0, model_cfg.vocab_size, (batch_size, model_cfg.context_length), device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    for _ in range(5):  # warmup
        with autocast_fn:
            _, loss, _ = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    sync()

    t0 = time.perf_counter()
    for _ in range(steps):
        with autocast_fn:
            _, loss, _ = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    sync()
    elapsed = time.perf_counter() - t0

    tokens = steps * batch_size * model_cfg.context_length
    return tokens / elapsed, elapsed / steps


def main() -> None:
    device = pick_device()
    print(f"device: {device}\n")

    configs = {
        "small (5.8M params, d_model=256)": ModelConfig(
            vocab_size=4096, context_length=256, d_model=256, n_layers=6, n_heads=8, d_ff=1024
        ),
        "larger (44M params, d_model=768)": ModelConfig(
            vocab_size=4096, context_length=256, d_model=768, n_layers=8, n_heads=12, d_ff=3072
        ),
    }

    for name, cfg in configs.items():
        n_params = MiniLLM(cfg).num_parameters()
        print(f"--- {name} ({n_params:,} params) ---", flush=True)
        for batch_size in (16, 32):
            print(f"  running batch={batch_size} fp32...", flush=True)
            fp32_tps, fp32_ms = bench(cfg, batch_size, mixed_precision=False, device=device)
            print(f"  running batch={batch_size} amp(bf16)...", flush=True)
            amp_tps, amp_ms = bench(cfg, batch_size, mixed_precision=True, device=device)
            speedup = (fp32_ms - amp_ms) / fp32_ms * 100
            print(
                f"  batch={batch_size:3d}  fp32: {fp32_tps:9,.0f} tok/s ({fp32_ms * 1000:6.1f} ms/step)  "
                f"amp(bf16): {amp_tps:9,.0f} tok/s ({amp_ms * 1000:6.1f} ms/step)  "
                f"delta: {speedup:+.1f}%",
                flush=True,
            )
        print(flush=True)


if __name__ == "__main__":
    main()
