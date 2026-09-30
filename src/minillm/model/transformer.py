from __future__ import annotations

import torch
import torch.nn as nn

from minillm.config import ModelConfig
from minillm.model.attention import CausalSelfAttention
from minillm.model.embeddings import TokenAndPositionalEmbedding

KVCache = tuple[torch.Tensor, torch.Tensor]


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class MiniLLM(nn.Module):
    """A decoder-only Transformer language model (GPT-style)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = TokenAndPositionalEmbedding(
            cfg.vocab_size, cfg.context_length, cfg.d_model, cfg.dropout
        )
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.embed.token_emb.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections, as in GPT-2, to keep the
        # variance of activations stable as depth grows.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed.pos_emb.weight.numel()
            if not self.cfg.tie_weights:
                n -= self.embed.token_emb.weight.numel()
        return n

    def forward(
        self,
        ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[KVCache | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[KVCache | None] | None]:
        """
        Args:
            ids: (batch, seq_len) token ids for the *new* tokens only.
            targets: (batch, seq_len) next-token targets for computing loss.
            kv_caches: per-layer caches from a previous call (for incremental decoding).
            use_cache: whether to return updated per-layer caches.
        """
        position_offset = kv_caches[0][0].size(2) if kv_caches and kv_caches[0] is not None else 0
        x = self.embed(ids, position_offset=position_offset)

        new_caches: list[KVCache | None] | None = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, updated = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(updated)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss, new_caches
