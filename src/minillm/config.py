from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    context_length: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class TrainConfig:
    batch_size: int = 64
    micro_batch_size: int | None = None  # for gradient accumulation; None = batch_size
    max_steps: int = 3000
    eval_interval: int = 250
    eval_iters: int = 50
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    mixed_precision: bool = True
    amp_dtype: str = "bf16"  # "bf16" or "fp16"
    seed: int = 1337
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500
    keep_last_n_checkpoints: int = 3
    log_interval: int = 20
