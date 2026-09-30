from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional KV-cache support.

    During training/full-sequence forward passes, a causal mask ensures
    position ``i`` can only attend to positions ``<= i``. During incremental
    decoding, a running key/value cache lets us attend over past tokens
    without recomputing their projections.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            x: (batch, seq_len, d_model) — new tokens only (seq_len == 1 during
               incremental decoding once a cache is present).
            kv_cache: optional (past_k, past_v), each (batch, n_heads, past_len, d_head).
            use_cache: whether to return the updated cache.
        Returns:
            (output, new_kv_cache)
        """
        b, t, c = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)  # (b, h, t, dh)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # When there's no cache (or we're processing the full prompt), the
        # query length equals the key length and we need a causal mask. Once
        # decoding incrementally with a cache, each forward pass only feeds
        # the single new query token, which may attend to *all* cached keys
        # (they're all in the past by construction) — no mask needed there.
        is_causal = q.size(2) == k.size(2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.resid_dropout(self.out_proj(out))
        return out, new_cache


def reference_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Slow, explicit (non-fused) causal attention — used only in tests to
    verify that no query position attends to a future key position."""
    d_head = q.size(-1)
    t_q, t_k = q.size(-2), k.size(-2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)
    causal_mask = torch.triu(
        torch.ones(t_q, t_k, dtype=torch.bool, device=q.device), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v
