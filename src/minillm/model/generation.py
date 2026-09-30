from __future__ import annotations

import torch
import torch.nn.functional as F

from minillm.model.transformer import MiniLLM


def apply_top_k_top_p(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth_value, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model: MiniLLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    eos_token_id: int | None = None,
    use_kv_cache: bool = True,
) -> torch.Tensor:
    """Autoregressive sampling.

    With ``use_kv_cache=True``, the prompt is processed once and each
    subsequent step only runs the model on the single newest token, reusing
    cached keys/values for every earlier position — this is what makes
    incremental decoding roughly O(n) instead of O(n^2) in sequence length.
    """
    model.eval()
    device = prompt_ids.device
    b = prompt_ids.size(0)
    generated = prompt_ids

    # This simple architecture has no sliding-window attention or rotary
    # position extrapolation, so absolute positions are only valid up to
    # `context_length`. Cap generation there rather than crashing on an
    # out-of-range positional embedding lookup once the cache fills up.
    max_new_tokens = min(max_new_tokens, max(0, model.cfg.context_length - prompt_ids.size(1)))
    if max_new_tokens == 0:
        return generated

    if use_kv_cache:
        ctx = model.cfg.context_length
        logits, _, kv_caches = model(prompt_ids[:, -ctx:], use_cache=True)
        next_logits = logits[:, -1, :]
    else:
        kv_caches = None
        logits, _, _ = model(generated[:, -model.cfg.context_length :])
        next_logits = logits[:, -1, :]

    finished = torch.zeros(b, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        scaled = next_logits / max(temperature, 1e-5)
        scaled = apply_top_k_top_p(scaled, top_k, top_p)
        probs = F.softmax(scaled, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if eos_token_id is not None:
            next_token = torch.where(
                finished.unsqueeze(-1), torch.full_like(next_token, eos_token_id), next_token
            )
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and bool(finished.all()):
            break

        if use_kv_cache:
            logits, _, kv_caches = model(next_token, kv_caches=kv_caches, use_cache=True)
        else:
            ctx = model.cfg.context_length
            logits, _, _ = model(generated[:, -ctx:])
        next_logits = logits[:, -1, :]

    return generated
