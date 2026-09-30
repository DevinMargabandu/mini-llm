from __future__ import annotations

import pytest
import torch

from minillm.config import ModelConfig
from minillm.model.transformer import MiniLLM


@pytest.fixture()
def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        context_length=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )


@pytest.fixture()
def tiny_model(tiny_config: ModelConfig) -> MiniLLM:
    torch.manual_seed(0)
    model = MiniLLM(tiny_config)
    model.eval()
    return model
