from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from minillm.config import ModelConfig
from minillm.model.transformer import MiniLLM
from minillm.serve.inference import ContinuousBatchingEngine
from minillm.serve.schemas import GenerateRequest, GenerateResponse, HealthResponse
from minillm.tokenizer.bpe import BPETokenizer

STATIC_DIR = Path(__file__).parent / "static"

state: dict = {}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(checkpoint_dir: str, device: torch.device):
    tokenizer = BPETokenizer.load(f"{checkpoint_dir}/tokenizer.json")
    ckpt = torch.load(f"{checkpoint_dir}/latest.pt", map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = MiniLLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_dir = os.environ.get("MINILLM_CHECKPOINT_DIR", "checkpoints")
    max_batch_size = int(os.environ.get("MINILLM_MAX_BATCH_SIZE", "16"))
    device = pick_device()
    model, tokenizer, cfg = load_model_and_tokenizer(checkpoint_dir, device)
    engine = ContinuousBatchingEngine(
        model, tokenizer, device, max_batch_size=max_batch_size
    )
    engine.start()
    state.update(model=model, tokenizer=tokenizer, cfg=cfg, device=device, engine=engine)
    yield
    await engine.stop()


app = FastAPI(title="MiniLLM", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(state["device"]),
        model_parameters=state["model"].num_parameters(),
        vocab_size=state["cfg"].vocab_size,
    )


@app.get("/stats")
def stats() -> dict:
    return state["engine"].stats


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest) -> GenerateResponse:
    if "engine" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")

    t0 = time.perf_counter()
    result = await state["engine"].submit(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    return GenerateResponse(
        prompt=req.prompt,
        completion=result["completion"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        latency_ms=latency_ms,
    )
