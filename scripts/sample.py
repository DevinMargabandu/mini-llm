#!/usr/bin/env python3
"""Quick CLI to sample from a trained checkpoint."""
from __future__ import annotations

import argparse

import torch

from minillm.config import ModelConfig
from minillm.model.generation import generate
from minillm.model.transformer import MiniLLM
from minillm.tokenizer.bpe import BPETokenizer
from minillm.train.train import pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = pick_device()
    tokenizer = BPETokenizer.load(f"{args.checkpoint_dir}/tokenizer.json")
    ckpt = torch.load(f"{args.checkpoint_dir}/latest.pt", map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = MiniLLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    out = generate(model, prompt_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
