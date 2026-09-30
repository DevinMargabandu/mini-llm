#!/usr/bin/env python3
"""Measure the real effect of (a) KV caching on single-request decoding
latency/throughput, and (b) continuous batching on multi-request serving
throughput, against the actual MiniLLM model + engine implementation.

Both benchmarks explicitly synchronize the device (torch.mps.synchronize /
torch.cuda.synchronize) around the timed region. Without that, wall-clock
timing on an async accelerator backend mostly measures how fast the CPU can
*enqueue* work rather than how long it actually takes to run, which silently
produces misleading numbers -- this bit us once while building this script,
so it's worth calling out explicitly rather than quietly fixing it.
"""
from __future__ import annotations

import asyncio
import time

import torch

from minillm.config import ModelConfig
from minillm.model.generation import generate
from minillm.model.transformer import MiniLLM
from minillm.serve.inference import ContinuousBatchingEngine
from minillm.tokenizer.bpe import BPETokenizer
from minillm.train.train import pick_device


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench_kv_cache(model: MiniLLM, device: torch.device, new_tokens: int, trials: int = 1):
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 16), device=device)

    def run(use_cache: bool) -> float:
        best = float("inf")
        for _ in range(trials):
            sync(device)
            t0 = time.perf_counter()
            generate(model, prompt, max_new_tokens=new_tokens, use_kv_cache=use_cache, top_k=None, top_p=None)
            sync(device)
            best = min(best, time.perf_counter() - t0)
        return best

    generate(model, prompt, max_new_tokens=8, use_kv_cache=True, top_k=None, top_p=None)  # warmup
    generate(model, prompt, max_new_tokens=8, use_kv_cache=False, top_k=None, top_p=None)
    sync(device)

    no_cache_t = run(False)
    cache_t = run(True)
    return no_cache_t, cache_t


async def bench_continuous_batching(
    model: MiniLLM, tokenizer: BPETokenizer, device: torch.device, n_requests: int, new_tokens: int
):
    prompt_ids = tokenizer.encode("To be, or not to be, that is the question:")

    sync(device)
    t0 = time.perf_counter()
    for _ in range(n_requests):
        generate(
            model,
            torch.tensor([prompt_ids], device=device),
            max_new_tokens=new_tokens,
            top_k=None,
            top_p=None,
        )
    sync(device)
    sequential_time = time.perf_counter() - t0

    engine = ContinuousBatchingEngine(model, tokenizer, device, max_batch_size=n_requests, tick_seconds=0.0)
    engine.start()
    sync(device)
    t0 = time.perf_counter()
    await asyncio.gather(
        *[
            engine.submit(
                "To be, or not to be, that is the question:",
                max_new_tokens=new_tokens,
                top_k=None,
                top_p=None,
            )
            for _ in range(n_requests)
        ]
    )
    sync(device)
    batched_time = time.perf_counter() - t0
    await engine.stop()

    return sequential_time, batched_time


def main() -> None:
    device = pick_device()
    print(f"device: {device}\n")

    tokenizer = BPETokenizer()
    tokenizer.train("to be or not to be that is the question " * 50, vocab_size=300)
    tokenizer.add_special_tokens(["<|endoftext|>"])
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size, context_length=1024, d_model=256, n_layers=6, n_heads=8, d_ff=1024
    )
    torch.manual_seed(0)
    model = MiniLLM(cfg).to(device).eval()

    print("--- KV cache vs. no cache, single request (sweeping generation length) ---")
    print(f"{'new_tokens':>10}  {'no_cache (ms)':>14}  {'cache (ms)':>12}  {'speedup':>8}")
    for new_tokens in (64, 256, 512, 900):
        no_cache_t, cache_t = bench_kv_cache(model, device, new_tokens=new_tokens)
        print(
            f"{new_tokens:>10}  {no_cache_t * 1000:>14.1f}  {cache_t * 1000:>12.1f}  "
            f"{no_cache_t / cache_t:>7.2f}x"
        )
    print(
        "\nNote: without a cache, every step reprocesses the whole (growing) sequence -- "
        "roughly O(n^2) total work over a generation. With a cache, each step only does "
        "O(1) work for the new token. For short generations, fixed per-step Python/kernel-"
        "dispatch overhead and cache-concatenation bookkeeping outweigh that saving; the "
        "crossover in the table above is where the O(n^2) cost catches up.\n"
    )

    print("--- Continuous batching vs. sequential (concurrent requests, 64 new tokens each) ---")
    print(f"{'n_requests':>10}  {'sequential (ms)':>16}  {'batched (ms)':>13}  {'speedup':>8}")
    for n_requests in (2, 4, 8, 16, 32):
        sequential_time, batched_time = asyncio.run(
            bench_continuous_batching(model, tokenizer, device, n_requests=n_requests, new_tokens=64)
        )
        print(
            f"{n_requests:>10}  {sequential_time * 1000:>16.1f}  {batched_time * 1000:>13.1f}  "
            f"{sequential_time / batched_time:>7.2f}x"
        )


if __name__ == "__main__":
    main()
