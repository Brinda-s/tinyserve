# TinyServe

A small LLM inference engine, built step by step to understand what serving systems like
vLLM actually do. Each step adds one idea, is checked against the official Hugging Face
implementation, and is benchmarked.

Model: **Qwen2.5-0.5B-Instruct** (494M parameters, 24 layers, grouped-query attention with
14 query heads and 2 KV heads, head dim 64). The model is written from scratch in plain
PyTorch in [`tinyserve/model.py`](tinyserve/model.py); only the tokenizer comes from
`transformers`.

## Roadmap

| Step | What | Status |
|---|---|---|
| 1 | Model from scratch + naive generation (recompute the whole sequence every token) | ✅ done |
| 2 | **KV cache**: prefill once, then decode one token at a time | ✅ done |
| 3 | Swap in the hand-written FlashAttention CUDA kernel from [gpu-kernels](https://github.com/Brinda-s/gpu-kernels) | next |
| 4 | **Continuous batching**: many requests at once, new ones join mid-generation | |
| 5 | **Paged KV cache**: memory in fixed-size blocks, like vLLM's PagedAttention | |
| 6 | Load test: throughput and latency as concurrent users grow from 1 to 64 | |

## What a KV cache does

To generate token *t*, attention needs the keys and values of every earlier token. Without a
cache, each step re-runs all 24 layers over the whole sequence, so step *t* costs about *t*
times as much as the first decode step. With a cache:

- **Prefill:** one pass over the prompt computes and stores K and V for every prompt token.
  This is compute-heavy and parallel, and it sets the *time to first token*.
- **Decode:** each step feeds only the newest token, appends its K and V to the cache, and
  attends over everything stored. Each step now reads all ~1 GB of weights to produce one
  token, so decode is **memory-bandwidth-bound**, the same regime as the softmax and
  LayerNorm kernels in gpu-kernels.

The cost is memory. For this model, per token:
`24 layers × 2 (K and V) × 2 KV heads × 64 dims × 2 bytes (fp16) = 12 KB`.
Grouped-query attention is why that's small: with 14 KV heads it would be 7× larger.

## Correctness

`tests/test_vs_hf.py` loads the same checkpoint into Hugging Face's own model and checks:

1. prompt logits match at every position
2. greedy generation picks identical tokens: our no-cache loop, our KV-cache loop and HF `generate()`
3. KV-cache logits match no-cache logits at every step
4. feeding the prompt in two chunks matches feeding it at once (tests the offset causal mask)

Result on CPU, fp32: logits identical to Hugging Face (max difference **0.0**), and **48/48**
greedy tokens identical across all three generators.

One thing the test caught: Qwen's instruct checkpoint ships `repetition_penalty: 1.1` in its
generation config, which Hugging Face applies even in greedy mode. The test turns it off so
both sides do plain greedy decoding.

## Results

Single request, 64 new tokens, Apple M-series CPU, fp32 (one run each, so expect noise):

| Prompt | Method | Time to first token | Decode tokens/s |
|---|---|---|---|
| 128 | no cache | 307 ms | 2.7 |
| 128 | **KV cache** | 197 ms | **10.1** (3.7×) |
| 128 | HF generate | 219 ms | 9.7 |
| 512 | no cache | 659 ms | 1.5 |
| 512 | **KV cache** | 601 ms | **13.9** (9×) |
| 512 | HF generate | 597 ms | 8.2 |

The no-cache loop gets slower as the prompt grows, because every step redoes the whole prompt.
The cached loop doesn't. GPU numbers (Colab T4) are next.

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m tests.test_vs_hf                    # correctness, CPU fp32
.venv/bin/python -m bench.bench_generate                 # benchmark (auto-picks cuda > mps > cpu)
.venv/bin/python -m tinyserve.chat "What is a KV cache?"
```

On a GPU (e.g. Colab):

```bash
pip install -r requirements.txt
python -m tests.test_vs_hf --device cuda --dtype float16
python -m bench.bench_generate --prompt-lens 128 512 1024 --new-tokens 128
```

The first run downloads the model (~1 GB) from the Hugging Face Hub.

## Layout

```
tinyserve/config.py     model hyper-parameters from config.json
tinyserve/model.py      Qwen2 decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU MLP
tinyserve/kv_cache.py   contiguous per-layer KV cache
tinyserve/loader.py     download from the Hub and load safetensors weights
tinyserve/generate.py   no-cache and KV-cache generation loops, with timing
tinyserve/chat.py       one-shot command-line chat
tests/test_vs_hf.py     correctness against Hugging Face
bench/bench_generate.py TTFT and decode throughput benchmark
```
