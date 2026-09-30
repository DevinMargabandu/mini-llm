#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f data/tinyshakespeare.txt ]; then
    python scripts/download_data.py
fi

if [ ! -f checkpoints/tokenizer.json ]; then
    python scripts/train_tokenizer.py
fi

python -m minillm.train.train "$@"
