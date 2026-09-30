from __future__ import annotations

import torch

from minillm.model.attention import CausalSelfAttention, reference_causal_attention


def test_fused_attention_matches_reference_causal_attention():
    torch.manual_seed(0)
    b, h, t, dh = 2, 3, 5, 8
    q = torch.randn(b, h, t, dh)
    k = torch.randn(b, h, t, dh)
    v = torch.randn(b, h, t, dh)

    expected = reference_causal_attention(q, k, v)
    actual = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_causal_mask_blocks_future_tokens():
    """Changing a future token must not change an earlier position's output."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=16, n_heads=4, dropout=0.0).eval()
    x = torch.randn(1, 6, 16)

    out1, _ = attn(x)

    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 100.0  # only touch the last (future-most) token
    out2, _ = attn(x_perturbed)

    # All positions except the last must be unaffected by the perturbation.
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)
    # The perturbed position itself should change.
    assert not torch.allclose(out1[:, -1, :], out2[:, -1, :])


def test_kv_cache_produces_identical_output_to_full_forward():
    """Incremental decoding with a KV cache must match a full-sequence pass."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=16, n_heads=4, dropout=0.0).eval()
    x = torch.randn(1, 6, 16)

    full_out, _ = attn(x)

    # Feed the sequence one token at a time, reusing the growing cache.
    cache = None
    outs = []
    for t in range(x.size(1)):
        out, cache = attn(x[:, t : t + 1, :], kv_cache=cache, use_cache=True)
        outs.append(out)
    incremental_out = torch.cat(outs, dim=1)

    assert torch.allclose(full_out, incremental_out, atol=1e-5)


def test_attention_output_shape():
    attn = CausalSelfAttention(d_model=32, n_heads=4, dropout=0.0)
    x = torch.randn(3, 10, 32)
    out, cache = attn(x, use_cache=True)
    assert out.shape == (3, 10, 32)
    assert cache[0].shape == (3, 4, 10, 8)
    assert cache[1].shape == (3, 4, 10, 8)
