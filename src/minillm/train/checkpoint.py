from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from minillm.config import ModelConfig


def save_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_cfg: ModelConfig,
    extra: dict[str, Any] | None = None,
    keep_last_n: int = 3,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / f"ckpt_step{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_cfg.__dict__,
            "extra": extra or {},
        },
        path,
    )

    latest = checkpoint_dir / "latest.pt"
    torch.save(torch.load(path, map_location="cpu", weights_only=False), latest)

    _prune_old_checkpoints(checkpoint_dir, keep_last_n)
    return path


def _prune_old_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    ckpts = sorted(
        checkpoint_dir.glob("ckpt_step*.pt"),
        key=lambda p: int(re.search(r"step(\d+)", p.stem).group(1)),
    )
    for stale in ckpts[:-keep_last_n] if keep_last_n > 0 else []:
        stale.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path, map_location: str = "cpu"
) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
