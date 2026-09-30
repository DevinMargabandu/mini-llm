from __future__ import annotations

import os

import pytest
import torch
from fastapi.testclient import TestClient

from minillm.config import ModelConfig
from minillm.model.transformer import MiniLLM
from minillm.tokenizer.bpe import BPETokenizer
from minillm.train.checkpoint import save_checkpoint


@pytest.fixture()
def checkpoint_dir(tmp_path):
    # A byte-level BPE tokenizer needs vocab_size >= 256 just to cover raw
    # bytes, so it can't share the 64-vocab `tiny_config` fixture used by the
    # shape/gradient tests (those feed the model random ids directly and
    # never go through a real tokenizer). Build a small but tokenizer-sized
    # model instead.
    tokenizer = BPETokenizer()
    tokenizer.train("hello world hello there general kenobi " * 20, vocab_size=280)
    tokenizer.add_special_tokens(["<|endoftext|>"])
    tokenizer.save(tmp_path / "tokenizer.json")

    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )
    torch.manual_seed(0)
    model = MiniLLM(model_cfg).eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(tmp_path, step=0, model=model, optimizer=optimizer, model_cfg=model_cfg)
    return tmp_path


@pytest.fixture()
def client(checkpoint_dir, monkeypatch):
    monkeypatch.setenv("MINILLM_CHECKPOINT_DIR", str(checkpoint_dir))
    # Force CPU for the API tests: the tiny fixture model doesn't need GPU
    # acceleration, and forcing CPU avoids contending with any real training
    # run that might be using the MPS/CUDA device at the same time.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    from minillm.serve.app import app

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_parameters"] > 0


def test_generate_endpoint_returns_completion(client):
    resp = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prompt"] == "hello"
    assert isinstance(body["completion"], str)
    assert body["completion_tokens"] <= 5
    assert body["latency_ms"] >= 0


def test_generate_endpoint_validates_request(client):
    resp = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 0})
    assert resp.status_code == 422


def test_concurrent_requests_are_all_served(client):
    import concurrent.futures

    def call(i):
        return client.post("/generate", json={"prompt": f"req{i}", "max_new_tokens": 4})

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        responses = list(pool.map(call, range(5)))

    assert all(r.status_code == 200 for r in responses)
