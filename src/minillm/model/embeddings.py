from __future__ import annotations

import torch
import torch.nn as nn


class TokenAndPositionalEmbedding(nn.Module):
    """Learned token embeddings + learned absolute positional embeddings."""

    def __init__(self, vocab_size: int, context_length: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.context_length = context_length

    def forward(self, ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        b, t = ids.shape
        positions = torch.arange(
            position_offset, position_offset + t, device=ids.device
        ).unsqueeze(0)
        x = self.token_emb(ids) + self.pos_emb(positions)
        return self.dropout(x)
