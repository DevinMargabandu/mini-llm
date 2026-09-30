from __future__ import annotations

import torch

from minillm.config import ModelConfig
from minillm.model.transformer import MiniLLM


def test_forward_logits_shape(tiny_model: MiniLLM, tiny_config: ModelConfig):
    b, t = 4, 10
    ids = torch.randint(0, tiny_config.vocab_size, (b, t))
    logits, loss, cache = tiny_model(ids)
    assert logits.shape == (b, t, tiny_config.vocab_size)
    assert loss is None
    assert cache is None


def test_forward_with_targets_returns_scalar_loss(tiny_model: MiniLLM, tiny_config: ModelConfig):
    b, t = 4, 10
    ids = torch.randint(0, tiny_config.vocab_size, (b, t))
    targets = torch.randint(0, tiny_config.vocab_size, (b, t))
    _, loss, _ = tiny_model(ids, targets=targets)
    assert loss.shape == ()
    assert loss.item() > 0


def test_kv_cache_shapes_grow_with_each_step(tiny_model: MiniLLM, tiny_config: ModelConfig):
    b = 2
    ids = torch.randint(0, tiny_config.vocab_size, (b, 3))
    logits, _, caches = tiny_model(ids, use_cache=True)
    assert logits.shape == (b, 3, tiny_config.vocab_size)
    assert len(caches) == tiny_config.n_layers
    for k, v in caches:
        assert k.shape == (b, tiny_config.n_heads, 3, tiny_config.d_head)
        assert v.shape == (b, tiny_config.n_heads, 3, tiny_config.d_head)

    next_token = torch.randint(0, tiny_config.vocab_size, (b, 1))
    logits2, _, caches2 = tiny_model(next_token, kv_caches=caches, use_cache=True)
    assert logits2.shape == (b, 1, tiny_config.vocab_size)
    for k, v in caches2:
        assert k.shape == (b, tiny_config.n_heads, 4, tiny_config.d_head)


def test_context_length_boundary_does_not_crash(tiny_config: ModelConfig, tiny_model: MiniLLM):
    ids = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.context_length))
    logits, _, _ = tiny_model(ids)
    assert logits.shape == (1, tiny_config.context_length, tiny_config.vocab_size)


def test_weight_tying_shares_storage():
    cfg = ModelConfig(
        vocab_size=32, context_length=8, d_model=16, n_layers=1, n_heads=2, d_ff=32, tie_weights=True
    )
    model = MiniLLM(cfg)
    assert model.lm_head.weight.data_ptr() == model.embed.token_emb.weight.data_ptr()
