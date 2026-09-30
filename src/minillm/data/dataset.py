from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from minillm.tokenizer.bpe import BPETokenizer


class TokenizedTextDataset(Dataset):
    """Wraps a flat 1-D array of token ids and yields fixed-length
    (input, target) chunks for next-token-prediction training."""

    def __init__(self, token_ids: np.ndarray, context_length: int) -> None:
        if len(token_ids) <= context_length:
            raise ValueError("corpus is shorter than one context window")
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.token_ids[idx : idx + self.context_length + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def build_token_arrays(
    text: str,
    tokenizer: BPETokenizer,
    val_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    split = int(len(ids) * (1 - val_fraction))
    return ids[:split], ids[split:]


def load_or_build_token_cache(
    text_path: str | Path,
    tokenizer: BPETokenizer,
    cache_dir: str | Path,
    val_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "train.bin"
    val_cache = cache_dir / "val.bin"

    if train_cache.exists() and val_cache.exists():
        return (
            np.fromfile(train_cache, dtype=np.uint16),
            np.fromfile(val_cache, dtype=np.uint16),
        )

    text = Path(text_path).read_text(encoding="utf-8")
    train_ids, val_ids = build_token_arrays(text, tokenizer, val_fraction)
    train_ids.tofile(train_cache)
    val_ids.tofile(val_cache)
    return train_ids, val_ids
