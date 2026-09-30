#!/usr/bin/env python3
"""Train the custom byte-level BPE tokenizer on a text corpus."""
from __future__ import annotations

import argparse
from pathlib import Path

from minillm.tokenizer.bpe import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", default="data/tinyshakespeare.txt")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--out", default="checkpoints/tokenizer.json")
    args = parser.parse_args()

    text = Path(args.text_path).read_text(encoding="utf-8")
    tok = BPETokenizer()
    tok.train(text, vocab_size=args.vocab_size, verbose=True)
    tok.add_special_tokens(["<|endoftext|>"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    print(f"Saved tokenizer ({tok.vocab_size} tokens) to {args.out}")

    sample = text[:200]
    ids = tok.encode(sample)
    roundtrip = tok.decode(ids)
    print(f"round-trip ok: {roundtrip == sample}")
    print(f"compression: {len(sample)} chars -> {len(ids)} tokens")


if __name__ == "__main__":
    main()
