#!/usr/bin/env python3
"""Download the TinyShakespeare corpus used for the demo training run.

This is the same public-domain text (the complete works of Shakespeare,
~1.1M characters) used by Karpathy's char-rnn / nanoGPT tutorials — small
enough to train a toy model on in minutes on a laptop CPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import requests

DEFAULT_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="data/tinyshakespeare.txt")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(args.url, timeout=30)
    resp.raise_for_status()
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"Wrote {len(resp.text):,} characters to {out_path}")


if __name__ == "__main__":
    main()
