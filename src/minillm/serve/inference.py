"""A small continuous-batching inference engine.

The core idea (borrowed from systems like Orca/vLLM, minus their paged
memory allocator): instead of running one request through the model at a
time, or waiting to accumulate a full static batch before starting, the
engine advances *all* in-flight requests one decoding step per iteration,
and lets new requests join and finished ones leave between iterations. That
keeps the GPU/CPU busy with batched matmuls instead of one tiny matmul per
request.

Simplification versus production systems: rather than padding+masking
sequences of different lengths into one tensor, requests are grouped by
their current KV-cache length each iteration (all requests admitted in the
same round advance in lock-step and naturally end up in the same group).
Each group is processed as one batched forward pass. This avoids needing a
padding attention mask, at the cost of only batching together requests
whose cache lengths currently coincide.
"""
from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field

import torch

from minillm.model.generation import apply_top_k_top_p
from minillm.model.transformer import KVCache, MiniLLM
from minillm.tokenizer.bpe import BPETokenizer

_ids = itertools.count()


@dataclass
class _ActiveRequest:
    prompt_tokens: list[int]
    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None
    future: asyncio.Future
    request_id: int = field(default_factory=lambda: next(_ids))
    generated: list[int] = field(default_factory=list)
    cache: list[KVCache] | None = None
    next_input: torch.Tensor | None = None  # (1, 1) token to feed next
    device: torch.device | None = None

    @property
    def cache_len(self) -> int:
        return 0 if self.cache is None else self.cache[0][0].size(2)


class ContinuousBatchingEngine:
    def __init__(
        self,
        model: MiniLLM,
        tokenizer: BPETokenizer,
        device: torch.device,
        max_batch_size: int = 16,
        tick_seconds: float = 0.002,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.tick_seconds = tick_seconds

        self._pending: list[_ActiveRequest] = []
        self._active: list[_ActiveRequest] = []
        self._lock = asyncio.Lock()
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.stats = {"steps": 0, "tokens_generated": 0, "requests_served": 0}

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def submit(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
    ) -> dict:
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        ctx = self.model.cfg.context_length
        prompt_tokens = self.tokenizer.encode(prompt)[-ctx:]
        # No sliding-window attention here, so absolute positions only go up
        # to context_length — cap how many new tokens we'll ever try to
        # generate rather than crashing once the cache fills up.
        max_new_tokens = max(1, min(max_new_tokens, ctx - len(prompt_tokens)))
        req = _ActiveRequest(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            future=future,
        )
        async with self._lock:
            self._pending.append(req)
        self._wakeup.set()
        result = await future
        return result

    async def _run_loop(self) -> None:
        while True:
            try:
                await self._step()
            except Exception as exc:  # noqa: BLE001
                # A crash here must not silently strand callers awaiting a
                # future that will now never resolve — fail every in-flight
                # request instead, then keep the loop alive for new ones.
                for req in self._pending + self._active:
                    if not req.future.done():
                        req.future.set_exception(exc)
                self._pending.clear()
                self._active.clear()

            if not self._active and not self._pending:
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(self.tick_seconds)

    async def _step(self) -> None:
        # NOTE: `@torch.no_grad()` as a *decorator* on an `async def` does
        # not do what it looks like — it wraps the call that creates the
        # coroutine object, not the body that runs when it's awaited, so it
        # silently leaves autograd tracking on for the whole method. Using
        # it as an inline context manager instead correctly scopes it to
        # this coroutine's actual execution.
        with torch.no_grad():
            async with self._lock:
                room = self.max_batch_size - len(self._active)
                admitted, self._pending = self._pending[:room], self._pending[room:]

            if admitted:
                for req in admitted:
                    self._prefill(req)  # one forward pass per request (unbatched: prompts vary in length)
                self._record_tokens(admitted, [r.next_input for r in admitted])
                for req in admitted:
                    self.stats["tokens_generated"] += 1
                    if self._is_done(req):
                        self._finish(req)
                    else:
                        self._active.append(req)

            if not self._active:
                return

            groups: dict[int, list[_ActiveRequest]] = {}
            for req in self._active:
                groups.setdefault(req.cache_len, []).append(req)

            finished: list[_ActiveRequest] = []
            for _, group in groups.items():
                self._step_group(group)
                self.stats["steps"] += 1
                for req in group:
                    self.stats["tokens_generated"] += 1
                    if self._is_done(req):
                        finished.append(req)

            for req in finished:
                self._active.remove(req)
                self._finish(req)

    def _is_done(self, req: _ActiveRequest) -> bool:
        return (
            req.generated[-1] == self._eos_id()
            or len(req.generated) >= req.max_new_tokens
            or req.cache_len >= self.model.cfg.context_length
        )

    def _finish(self, req: _ActiveRequest) -> None:
        text = self.tokenizer.decode(req.generated)
        if not req.future.done():
            req.future.set_result(
                {
                    "completion": text,
                    "prompt_tokens": len(req.prompt_tokens),
                    "completion_tokens": len(req.generated),
                }
            )
        self.stats["requests_served"] += 1

    def _eos_id(self) -> int | None:
        return self.tokenizer.special_tokens.get("<|endoftext|>")

    def _prefill(self, req: _ActiveRequest) -> None:
        ctx = self.model.cfg.context_length
        ids = torch.tensor(
            [req.prompt_tokens[-ctx:]], dtype=torch.long, device=self.device
        )
        logits, _, cache = self.model(ids, use_cache=True)
        next_token = self._sample(logits[:, -1, :], req)
        req.cache = cache
        req.next_input = next_token

    def _step_group(self, group: list[_ActiveRequest]) -> None:
        batch_input = torch.cat([r.next_input for r in group], dim=0)  # (n, 1)
        n_layers = len(group[0].cache)
        batched_cache: list[KVCache] = []
        for layer in range(n_layers):
            ks = torch.cat([r.cache[layer][0] for r in group], dim=0)
            vs = torch.cat([r.cache[layer][1] for r in group], dim=0)
            batched_cache.append((ks, vs))

        logits, _, new_cache = self.model(batch_input, kv_caches=batched_cache, use_cache=True)

        next_tokens = [self._sample(logits[i : i + 1, -1, :], req) for i, req in enumerate(group)]
        for i, req in enumerate(group):
            req.cache = [(k[i : i + 1], v[i : i + 1]) for k, v in new_cache]
            req.next_input = next_tokens[i]
        self._record_tokens(group, next_tokens)

    def _record_tokens(self, group: list[_ActiveRequest], next_tokens: list[torch.Tensor]) -> None:
        # Reading each sampled token back to Python with .item() forces a
        # device sync; calling it once per request (once per decoding step,
        # per request) was the dominant cost in this engine. Concatenating
        # first means the whole group pays for exactly one sync per step
        # instead of `len(group)` of them.
        ids = torch.cat(next_tokens, dim=0).squeeze(-1).tolist()
        for req, token_id in zip(group, ids):
            req.generated.append(token_id)

    def _sample(self, logits: torch.Tensor, req: _ActiveRequest) -> torch.Tensor:
        scaled = logits / max(req.temperature, 1e-5)
        scaled = apply_top_k_top_p(scaled, req.top_k, req.top_p)
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1)
