from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: int | None = Field(default=50, ge=1)
    top_p: float | None = Field(default=0.95, gt=0.0, le=1.0)


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    device: str
    model_parameters: int
    vocab_size: int
