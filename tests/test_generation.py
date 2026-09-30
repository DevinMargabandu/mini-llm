from __future__ import annotations

import torch

from minillm.config import ModelConfig
from minillm.model.generation import generate
from minillm.model.transformer import MiniLLM


def test_generate_appends_requested_number_of_tokens(tiny_model: MiniLLM, tiny_config: ModelConfig):
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 5))
    out = generate(tiny_model, prompt, max_new_tokens=7, top_k=None, top_p=None)
    assert out.shape == (1, 5 + 7)
    assert torch.equal(out[:, :5], prompt)


def test_generate_supports_batched_prompts(tiny_model: MiniLLM, tiny_config: ModelConfig):
    prompt = torch.randint(0, tiny_config.vocab_size, (3, 4))
    out = generate(tiny_model, prompt, max_new_tokens=5)
    assert out.shape == (3, 4 + 5)


def test_kv_cache_generation_matches_no_cache_generation_greedily(
    tiny_model: MiniLLM, tiny_config: ModelConfig
):
    """With temperature -> 0 (near-greedy) and a fixed seed, cached and
    uncached decoding should pick the same tokens, since they compute the
    same logits at every step — the cache is purely a performance
    optimization, not an approximation."""
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))

    torch.manual_seed(42)
    out_cached = generate(
        tiny_model, prompt, max_new_tokens=6, temperature=1e-4, top_k=None, top_p=None, use_kv_cache=True
    )
    torch.manual_seed(42)
    out_uncached = generate(
        tiny_model, prompt, max_new_tokens=6, temperature=1e-4, top_k=None, top_p=None, use_kv_cache=False
    )
    assert torch.equal(out_cached, out_uncached)


def test_generation_stops_early_on_eos(tiny_config: ModelConfig):
    torch.manual_seed(0)
    model = MiniLLM(tiny_config)
    model.eval()
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 3))
    eos_id = 5
    out = generate(model, prompt, max_new_tokens=50, eos_token_id=eos_id)
    generated_part = out[0, 3:].tolist()
    if eos_id in generated_part:
        first_eos = generated_part.index(eos_id)
        # everything after the first EOS should also be EOS (padding behavior)
        assert all(tok == eos_id for tok in generated_part[first_eos:])
