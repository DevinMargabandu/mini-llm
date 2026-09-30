# MiniLLM

A decoder-only Transformer language model built from scratch in PyTorch — custom
byte-level BPE tokenizer, causal multi-head attention, mixed-precision training,
and a Dockerized FastAPI serving layer with KV-cached, continuously-batched
inference.

This is a small-scale (~6M parameter) model trained on the TinyShakespeare corpus,
built to demonstrate the full pipeline end-to-end — tokenizer → model → training →
serving — rather than to compete with a real production LLM.

## Architecture

- **Tokenizer** ([`tokenizer/bpe.py`](src/minillm/tokenizer/bpe.py)): byte-level BPE
  trained from scratch (no external tokenizer libraries), in the style of GPT-2 —
  text is split into word-ish chunks with a regex, then adjacent byte pairs are
  iteratively merged by frequency. Uses an incremental pair-count index so training
  on a ~1.1M character corpus takes ~2 seconds instead of the tens of minutes a
  naive full-rescan-per-merge implementation would take.
- **Model** ([`model/transformer.py`](src/minillm/model/transformer.py)): standard
  pre-norm GPT-style decoder — learned token + positional embeddings, causal
  multi-head self-attention (`torch.nn.functional.scaled_dot_product_attention`),
  GELU MLP blocks, weight-tied input/output embeddings.
- **KV-cached generation** ([`model/generation.py`](src/minillm/model/generation.py),
  [`model/attention.py`](src/minillm/model/attention.py)): each attention layer can
  return and accept a `(key, value)` cache so incremental decoding only computes
  projections for the newest token instead of recomputing the whole prefix.
- **Training** ([`train/train.py`](src/minillm/train/train.py)): AdamW, cosine LR
  schedule with linear warmup, gradient clipping, checkpointing (rolling
  keep-last-N), and `torch.autocast` mixed precision (bf16 on CPU/CUDA/MPS, fp16 +
  `GradScaler` on CUDA) with `torch.profiler` hooks (`--profile`) for
  identifying step-time bottlenecks.
- **Serving** ([`serve/app.py`](src/minillm/serve/app.py),
  [`serve/inference.py`](src/minillm/serve/inference.py)): a FastAPI app backed by
  a small continuous-batching engine — see [Continuous batching](#continuous-batching-design)
  below for how it actually works and where it's simplified relative to
  production systems like vLLM.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 1. Get the corpus (public-domain Shakespeare, ~1.1M characters)
python scripts/download_data.py

# 2. Train the BPE tokenizer (~2s)
python scripts/train_tokenizer.py --vocab-size 4096

# 3. Train the model (~9 min on an M-series Mac; see the note on AMP below)
python -m minillm.train.train --max-steps 1200 --batch-size 32 --no-amp

# 4. Serve it
uvicorn minillm.serve.app:app --port 8000
curl -X POST localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt": "ROMEO:", "max_new_tokens": 100}'
```

Or just `./scripts/train.sh` / `./scripts/serve.sh`.

### Sample output

The included `checkpoints/latest.pt` is the actual result of the command
above: a 5.8M-parameter model, 4096-token BPE vocabulary, trained for 1200
steps (~9 minutes) on TinyShakespeare. Train loss went from 8.38 (≈ln(4096),
i.e. random) down to 4.09; val loss to 4.53. That's a small model trained
briefly, not a converged one — enough to learn structure (character-name
headers, line breaks, capitalization, iambic-ish rhythm) without producing
fully coherent Shakespeare. `python scripts/sample.py --prompt "ROMEO:"`:

```
ROMEO:
I will be, the man that the mot of an tap,
And made my soul's salal to my pound,
And yet in my blood to be in thy sire
And then the ruthless nuriously and my body's face,
Which, but, my heart with my life, as I may.

ROMEO:
I'll do not so so to it in my breast
To keep my lord.
```

### Docker

```bash
docker build -t minillm .
docker run -p 8000:8000 minillm
```

## Testing

```bash
pytest -v
```

28 tests across [`tests/`](tests/) cover:
- **tokenizer round-trips** — training text, unseen text, unicode/emoji, empty
  string, special tokens, save/load ([`test_tokenizer.py`](tests/test_tokenizer.py))
- **attention masking correctness** — the fused SDPA kernel matches an explicit
  reference implementation, a perturbation to a future token provably doesn't
  change earlier positions' outputs, and KV-cached incremental decoding produces
  bit-identical results to a full-sequence forward pass
  ([`test_attention.py`](tests/test_attention.py))
- **tensor shapes** — logits/cache shapes through forward passes, incremental
  decoding, context-length boundaries, weight tying
  ([`test_shapes.py`](tests/test_shapes.py))
- **gradient flow** — every parameter receives a finite, non-zero gradient;
  gradient clipping actually bounds the norm; loss decreases over a few optimizer
  steps ([`test_gradients.py`](tests/test_gradients.py))
- **generation** — correct output length, batched prompts, cached vs. uncached
  decoding pick identical tokens (the cache is a speed optimization, not an
  approximation), early stopping on EOS
  ([`test_generation.py`](tests/test_generation.py))
- **the FastAPI service itself** — `/health`, `/generate` end-to-end against a
  real (tiny) checkpoint, request validation, and concurrent requests all
  completing successfully through the continuous-batching engine
  ([`test_api.py`](tests/test_api.py))

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the suite on
Python 3.11 and 3.12 on every push/PR, plus a Docker build sanity check.

## Continuous batching design

Production inference servers (vLLM, TensorRT-LLM, Orca) implement continuous
batching with a *paged* KV cache — physical memory is allocated in fixed-size
blocks so sequences of any length can be mixed in one batch without padding
waste. That block allocator is most of that class of system's engineering effort
and is out of scope here.

[`serve/inference.py`](src/minillm/serve/inference.py) implements the same
scheduling *idea* — new requests join and finished ones leave the active batch
between decoding iterations, instead of waiting for a static batch to fully
drain — with a simpler mechanism: requests are grouped by their current KV-cache
length each iteration (everything admitted in the same round advances in
lock-step and lands in the same group), and each group runs as one batched
forward pass. This gets the core throughput win — the GPU/CPU spends its time on
batched matmuls instead of one-request-at-a-time matmuls — without needing a
padding attention mask or a memory allocator. The tradeoff: requests only batch
with others admitted in the same round, not with arbitrary in-flight requests of
different lengths.

## Benchmarks

All numbers below were measured on this development machine (Apple Silicon,
`torch` MPS backend) — see [`benchmarks/`](benchmarks/) to reproduce them.
Absolute numbers depend heavily on hardware; the qualitative behavior (KV
caching wins, batching wins) is the reproducible part.

Getting trustworthy numbers here took two rounds of debugging real bugs in the
serving engine, which turned out to be more informative than the benchmark
itself — worth recounting briefly:

1. **A silently-broken `@torch.no_grad()`.** It was applied as a decorator on
   an `async def` method. That pattern doesn't do what it looks like: the
   decorator wraps the call that *creates* the coroutine object, not the
   coroutine body that actually runs later when it's awaited — so the context
   manager had already exited by the time any tensor op executed, and every
   decoding step was silently building autograd graphs it never needed.
   Fixed by using `with torch.no_grad():` as an inline block inside the
   method instead ([`serve/inference.py`](src/minillm/serve/inference.py)).
2. **A `.item()` call per request per step.** Reading a sampled token back to
   Python forces a device sync; doing that once per request in a batch (`n`
   syncs) rather than once per batch (`1` sync via a single `.tolist()` over
   the concatenated tokens) meant the "batched" engine was paying for its
   batching benefit with `n`x the sync overhead.

Both are the kind of bug that doesn't crash anything — the server works fine,
generates correct text, passes every test — it's just slower than it should
be, and only shows up under a timing microscope. See
[`benchmarks/inference_results.txt`](benchmarks/inference_results.txt) for the
full before/after story.

### KV cache vs. no cache (single request, sweeping generation length)

Run: `python benchmarks/benchmark_inference.py`

| new tokens | no cache | with cache | speedup |
|---:|---:|---:|---:|
| 64  | 108 ms   | 215 ms   | 0.50x |
| 256 | 416 ms   | 823 ms   | 0.51x |
| 512 | 1,200 ms | 1,285 ms | 0.93x |
| 900 | 3,485 ms | 2,603 ms | **1.34x** |

Without a cache, every step reprocesses the whole (growing) sequence — O(n²)
total work over a generation. With a cache, each step only does O(1) work for
the new token, at the cost of concatenating the cache tensors every step. For
short generations the fixed per-step overhead and that concatenation cost
outweigh the saving; the crossover above (between 512 and 900 tokens, on this
6M-parameter model on this hardware) is where the O(n²) cost catches up and
the cache starts winning. The lesson isn't "caching is always faster" — it's
*why* the crossover exists and where it sits for a given model/hardware.

### Continuous batching vs. sequential (concurrent requests, 64 tokens each)

Same script:

| concurrent requests | sequential | batched | speedup |
|---:|---:|---:|---:|
| 2  | 311 ms   | 509 ms   | 0.61x |
| 4  | 344 ms   | 367 ms   | 0.94x |
| 8  | 681 ms   | 658 ms   | 1.04x |
| 16 | 1,432 ms | 825 ms   | **1.74x** |
| 32 | 2,599 ms | 1,132 ms | **2.30x** |

Batching multiple in-flight requests together keeps the model doing one
batch-of-N matmul per step instead of N separate batch-of-1 matmuls; the
throughput win grows with concurrency because the fixed per-step overhead is
amortized across more requests. At low concurrency (2-4 requests) that fixed
overhead still dominates and batching isn't worth it yet — which is itself a
realistic result: production batching schedulers face the same tradeoff and
use queuing delays / minimum batch sizes to manage it.

### Mixed precision (`torch.autocast`)

Run: `python benchmarks/benchmark_training.py`

This is the one place where the honest result contradicts the usual
expectation, and it's worth stating plainly rather than papering over: **on
this machine's Apple Silicon MPS backend, `bfloat16` autocast was not faster
than fp32** for either the 5.8M or 60M-parameter configs tested — see
[`benchmarks/training_results.txt`](benchmarks/training_results.txt). The
well-documented ~20–30% mixed-precision speedup is a property of NVIDIA Tensor
Cores doing native fp16/bf16 matmuls at roughly double the fp32 throughput;
Apple's MPS backend doesn't currently expose that same hardware advantage for
this workload shape, so autocast there mostly just adds cast overhead. The
training code supports CUDA fp16 (with `GradScaler`) and CUDA/CPU/MPS bf16
identically — on CUDA hardware this is where you'd expect to reproduce the
classic speedup.

## Project structure

```
src/minillm/
  tokenizer/bpe.py       byte-level BPE tokenizer (train/encode/decode/save/load)
  model/
    attention.py         causal multi-head self-attention + KV cache
    embeddings.py         token + positional embeddings
    transformer.py        decoder block, full MiniLLM model
    generation.py          autoregressive sampling (temperature/top-k/top-p, KV cache)
  data/dataset.py         tokenized dataset + train/val split + on-disk caching
  train/
    train.py               training loop (AMP, cosine schedule, checkpointing, profiler)
    checkpoint.py            checkpoint save/load with rolling retention
  serve/
    app.py                  FastAPI app
    inference.py             continuous-batching engine
    schemas.py                request/response models
tests/                    pytest suite (attention, shapes, gradients, tokenizer, generation, API)
benchmarks/                throughput/latency measurement scripts
scripts/                    CLI entry points (download data, train tokenizer, train, serve)
.github/workflows/ci.yml   GitHub Actions: pytest on 3.11/3.12 + Docker build check
```

## Limitations

This is a portfolio/demo-scale project, not a production LLM:
- ~6M parameters, 4096-token BPE vocabulary, 256-token context window — trained
  on one small public-domain corpus (TinyShakespeare, ~1.1M characters).
- No sliding-window attention or position extrapolation, so a single generation
  is capped at the model's context length (see the docstring in
  [`generation.py`](src/minillm/model/generation.py)).
- The continuous-batching engine is a from-scratch teaching implementation of
  the scheduling idea, not a paged-attention memory manager — see
  [Continuous batching design](#continuous-batching-design).
