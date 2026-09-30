from __future__ import annotations

import torch

from minillm.config import ModelConfig
from minillm.model.transformer import MiniLLM


def test_all_parameters_receive_finite_nonzero_gradients(tiny_config: ModelConfig):
    torch.manual_seed(0)
    model = MiniLLM(tiny_config)
    model.train()

    b, t = 4, 12
    ids = torch.randint(0, tiny_config.vocab_size, (b, t))
    targets = torch.randint(0, tiny_config.vocab_size, (b, t))

    _, loss, _ = model(ids, targets=targets)
    loss.backward()

    zero_grad_params = []
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradients"
        if torch.all(p.grad == 0):
            zero_grad_params.append(name)

    assert not zero_grad_params, f"parameters with all-zero gradients: {zero_grad_params}"


def test_gradient_clipping_bounds_norm(tiny_config: ModelConfig):
    torch.manual_seed(0)
    model = MiniLLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    targets = torch.randint(0, tiny_config.vocab_size, (2, 8))

    _, loss, _ = model(ids, targets=targets)
    (loss * 1000).backward()  # inflate to guarantee the clip threshold is exceeded

    max_norm = 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    assert total_norm > max_norm  # sanity: clipping was actually necessary

    post_clip_norm = torch.sqrt(
        sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)
    )
    assert post_clip_norm <= max_norm + 1e-3


def test_loss_decreases_after_a_few_optimizer_steps(tiny_config: ModelConfig):
    torch.manual_seed(0)
    model = MiniLLM(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    ids = torch.randint(0, tiny_config.vocab_size, (4, 12))
    targets = torch.randint(0, tiny_config.vocab_size, (4, 12))

    losses = []
    for _ in range(20):
        _, loss, _ = model(ids, targets=targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
