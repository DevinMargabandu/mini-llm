#!/usr/bin/env python3
"""Training loop for MiniLLM.

Supports gradient accumulation, a cosine LR schedule with linear warmup,
periodic checkpointing, and mixed-precision training via ``torch.autocast``
(bf16 on CUDA/CPU, fp16 + GradScaler on CUDA when bf16 isn't requested).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from minillm.config import ModelConfig, TrainConfig
from minillm.data.dataset import TokenizedTextDataset, load_or_build_token_cache
from minillm.model.transformer import MiniLLM
from minillm.tokenizer.bpe import BPETokenizer
from minillm.train.checkpoint import save_checkpoint


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, train_cfg: TrainConfig):
    if not train_cfg.mixed_precision:
        return torch.autocast(device_type="cpu", enabled=False)
    if device.type == "cuda":
        dtype = torch.bfloat16 if train_cfg.amp_dtype == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu":
        # bf16 autocast is supported on CPU and is the only numerically safe
        # choice there (no GradScaler needed, unlike fp16).
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    if device.type == "mps":
        dtype = torch.bfloat16 if train_cfg.amp_dtype == "bf16" else torch.float16
        return torch.autocast(device_type="mps", dtype=dtype)
    return torch.autocast(device_type="cpu", enabled=False)


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_learning_rate
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + coeff * (cfg.learning_rate - cfg.min_learning_rate)


@torch.no_grad()
def estimate_loss(
    model: MiniLLM,
    loaders: dict[str, DataLoader],
    device: torch.device,
    eval_iters: int,
    autocast_fn,
) -> dict[str, float]:
    model.eval()
    out = {}
    for split, loader in loaders.items():
        losses = torch.zeros(eval_iters)
        it = iter(loader)
        for i in range(eval_iters):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            with autocast_fn:
                _, loss, _ = model(x, targets=y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def build_dataloaders(
    train_ids, val_ids, context_length: int, batch_size: int
) -> dict[str, DataLoader]:
    train_ds = TokenizedTextDataset(train_ids, context_length)
    val_ds = TokenizedTextDataset(val_ids, context_length)
    common = dict(batch_size=batch_size, shuffle=True, drop_last=True)
    return {
        "train": DataLoader(train_ds, num_workers=0, **common),
        "val": DataLoader(val_ds, num_workers=0, **common),
    }


def train(
    text_path: str,
    tokenizer_path: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    profile: bool = False,
) -> None:
    device = pick_device()
    print(f"Training on device: {device}")

    torch.manual_seed(train_cfg.seed)

    tokenizer = BPETokenizer.load(tokenizer_path)
    train_ids, val_ids = load_or_build_token_cache(
        text_path, tokenizer, cache_dir="data/tokenized"
    )
    print(f"train tokens: {len(train_ids):,}  val tokens: {len(val_ids):,}")

    loaders = build_dataloaders(train_ids, val_ids, model_cfg.context_length, train_cfg.batch_size)

    model = MiniLLM(model_cfg).to(device)
    print(f"model parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    use_fp16_scaler = (
        train_cfg.mixed_precision and device.type == "cuda" and train_cfg.amp_dtype == "fp16"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    autocast_fn = autocast_context(device, train_cfg)

    train_iter = iter(loaders["train"])
    model.train()

    t0 = time.time()
    running_loss = 0.0

    profiler_ctx = _build_profiler(profile)
    with profiler_ctx as prof:
        for step in range(train_cfg.max_steps):
            lr = get_lr(step, train_cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(loaders["train"])
                x, y = next(train_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with autocast_fn:
                _, loss, _ = model(x, targets=y)

            optimizer.zero_grad(set_to_none=True)
            if use_fp16_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()

            running_loss += loss.item()
            if prof is not None:
                prof.step()

            if step % train_cfg.log_interval == 0:
                elapsed = time.time() - t0
                avg_loss = running_loss / max(1, min(step + 1, train_cfg.log_interval))
                toks_per_sec = (
                    train_cfg.log_interval
                    * train_cfg.batch_size
                    * model_cfg.context_length
                    / max(elapsed, 1e-9)
                ) if step > 0 else 0.0
                print(
                    f"step {step:5d} | loss {avg_loss:.4f} | lr {lr:.2e} "
                    f"| {toks_per_sec:,.0f} tok/s"
                )
                running_loss = 0.0
                t0 = time.time()

            if step > 0 and step % train_cfg.eval_interval == 0:
                losses = estimate_loss(model, loaders, device, train_cfg.eval_iters, autocast_fn)
                print(f"  eval @ step {step}: train {losses['train']:.4f}  val {losses['val']:.4f}")

            if step > 0 and step % train_cfg.checkpoint_interval == 0:
                save_checkpoint(
                    train_cfg.checkpoint_dir,
                    step,
                    model,
                    optimizer,
                    model_cfg,
                    keep_last_n=train_cfg.keep_last_n_checkpoints,
                )

    save_checkpoint(
        train_cfg.checkpoint_dir,
        train_cfg.max_steps,
        model,
        optimizer,
        model_cfg,
        keep_last_n=train_cfg.keep_last_n_checkpoints,
    )
    Path(train_cfg.checkpoint_dir, "tokenizer.json").write_bytes(Path(tokenizer_path).read_bytes())
    print("Training complete.")


def _build_profiler(enabled: bool):
    if not enabled:
        import contextlib

        return contextlib.nullcontext(None)
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_traces"),
        record_shapes=True,
        with_stack=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", default="data/tinyshakespeare.txt")
    parser.add_argument("--tokenizer-path", default="checkpoints/tokenizer.json")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.no_amp:
        train_cfg.mixed_precision = False

    train(args.text_path, args.tokenizer_path, model_cfg, train_cfg, profile=args.profile)


if __name__ == "__main__":
    main()
