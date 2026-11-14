# TinyServe

A small LLM inference engine, built step by step to understand what serving systems like
vLLM actually do. Each step adds one idea, is checked for exact agreement with a reference,
and is benchmarked.

Model: **Qwen2.5-0.5B-Instruct** (494M parameters, 24 layers, grouped-query attention with
14 query heads sharing 2 KV heads, head dim 64). The model is written from scratch in plain
PyTorch in [`tinyserve/model.py`](tinyserve/model.py); only the tokenizer comes from
`transformers`.

## Roadmap

| Step | What | Status |
|---|---|---|
| 1 | Model from scratch + naive generation (recompute the whole sequence every token) | ✅ matches Hugging Face exactly |
| 2 | **KV cache**: prefill once, then decode one token at a time | ✅ 5.9× faster decode at 2k-token prompts (T4) |
| 3 | **Hand-written FlashAttention** from [gpu-kernels](https://github.com/Brinda-s/gpu-kernels) for prefill | ✅ correct in the full model; prefill 1.0–1.2× (see why below) |
| 4 | **Continuous batching**: requests join and leave the batch every step | ✅ token-identical to single requests |
| 5 | **Paged KV cache**: memory in 16-token blocks, with preemption when full | ✅ token-identical, incl. preemption |
| 6 | **Load test**: four serving strategies under the same memory budget | ✅ T4: **17×** the throughput of one-at-a-time serving |

## Results

### Step 6: serving 64 requests with the same 192 MB of KV-cache memory (Tesla T4, fp16)

64 requests arriving at once, prompts 32–512 tokens, outputs 16–256 tokens (8,523 output tokens).
Every configuration gets a KV-cache budget of 16,384 tokens: 21 slots of 768 tokens, or 1,024
blocks of 16.

| Strategy | Throughput | Wall time | Median time to first token | p95 latency | Mean batch | Peak concurrent | KV memory actually used |
|---|---|---|---|---|---|---|---|
| Sequential | 34 tok/s | 248.5 s | 132.1 s | 240.1 s | 1.0 | 1 | 92% |
| Static batching | 342 tok/s | 24.9 s | 8.0 s | 23.3 s | 11.0 | 21 | 42% |
| Continuous + slots | 506 tok/s | 16.8 s | 4.3 s | 15.6 s | 17.0 | 21 | 52% |
| **Continuous + paged** | **593 tok/s** | **14.4 s** | **0.97 s** | **13.5 s** | **31.1** | **54** | **98%** |

What the table shows:

- **One request at a time leaves the GPU idle.** A single sequence decodes at about 42 tokens/s on
  the T4 (see step 2 below), nowhere near what the hardware can do, so serving 64 people one by one
  takes four minutes.
- **Batching is almost free per extra request.** A decode step with 31 sequences costs little more
  than a step with one, because each step reads the same weights. That is why static batching is
  already 10× faster.
- **Static batching wastes the batch.** A batch ends only when its longest request does, so the mean
  batch (11.0) stays well below the 21 slots.
- **Continuous batching keeps the slots full**: +48% throughput, and half the time to first token.
- **Slots waste memory.** Each slot reserves 768 tokens for the longest possible request; only 52% of
  that holds real tokens, which caps the batch at 21.
- **Paged memory fixes that.** 98% of reserved memory holds real tokens, up to 54 requests run at
  once, throughput rises another 17%, and the median request sees its first token in under a second
  instead of 4.3 s. The 18 preemptions are the price: paused requests are recomputed later.

The same experiment on the CPU (24 requests, 48 MB budget) showed the same ordering: paged
continuous batching at 2.8× sequential and 2.1× slots.

### Step 3: the gpu-kernels FlashAttention inside the model (Tesla T4, fp16)

The kernel matches PyTorch SDPA to within 1.2e-3 for prompt lengths 1 to 1,031 (including lengths
that need padding), and with it plugged in the model still produces the same 32 greedy tokens as
Hugging Face. Speed, for the whole model's prefill:

| Prompt | SDPA | gpu-kernels FlashAttention | Speedup |
|---|---|---|---|
| 512 | 36.5 ms | 30.7 ms | 1.19× |
| 1,024 | 51.6 ms | 52.2 ms | 0.99× |
| 2,048 | 109.5 ms | 108.4 ms | 1.01× |
| 4,096 | 234.2 ms | 227.2 ms | 1.03× |

The kernel alone is 21–26% faster than PyTorch's fp16 attention on this GPU (gpu-kernels
benchmark), so why does the model barely speed up? **Amdahl's law.** For a 0.5B model, prefill time
is dominated by the linear layers (about 0.7 GFLOP per token). Attention is a small share: about 3%
of the work at 512 tokens and about 20% at 4,096. A 25% faster kernel applied to 20% of the work
gives at most a 5% gain, and padding to 64-token tiles plus fp16/fp32 conversions eat part of that.
Attention kernels pay off on long contexts and larger head counts, not on short prompts to a small
model.

### Step 2: KV cache, single request (Tesla T4, fp16, 128 new tokens)

| Prompt | Method | Time to first token | Decode tokens/s |
|---|---|---|---|
| 128 | no cache | 22.8 ms | 42.7 |
| 128 | **KV cache** | 27.1 ms | **43.1** |
| 512 | no cache | 29.6 ms | 29.9 |
| 512 | **KV cache** | 26.0 ms | **41.9** (1.4×) |
| 1,024 | no cache | 60.1 ms | 14.8 |
| 1,024 | **KV cache** | 52.5 ms | **42.6** (2.9×) |
| 2,048 | no cache | 130.0 ms | 7.1 |
| 2,048 | **KV cache** | 150.4 ms | **41.8** (5.9×) |
| 2,048 | HF generate | 117.7 ms | 40.1 |

Without a cache, decode slows as the context grows (42.7 → 7.1 tok/s), because every step
redoes the whole sequence. With the cache it stays flat at about 42 tok/s, slightly ahead of
Hugging Face. At a 128-token prompt the cache makes no difference on a GPU: recomputing 128 tokens
in parallel costs about the same as computing one.

That flat 42 tok/s is itself a finding. Reading ~1 GB of fp16 weights at the T4's 320 GB/s would
allow roughly 300 tokens/s, so single-request decode reaches only about 13% of the memory limit.
Each token runs about 24 layers × ~20 small GPU operations, and at batch size 1 the time goes to
launching them from Python, not to the work inside them. This is why batching (step 6) helps so
much, and why production engines capture the decode step in a CUDA graph.

### On the CPU (Apple silicon laptop, fp32)

KV cache, single request, 64 new tokens: 3.7× faster decode at a 128-token prompt and 9× at
512 tokens. Load test (24 requests, 48 MB budget): sequential 9.8 tok/s, static batching 11.1,
continuous + slots 13.0, continuous + paged 27.5 tok/s.

## Correctness

Every step is held to exact agreement, not "looks reasonable":

| Test | What it checks | Result (CPU fp32 unless noted) |
|---|---|---|
| `tests/test_vs_hf.py` | prompt logits vs the Hugging Face model | max difference **0.0** |
| | greedy tokens: our no-cache loop, our KV-cache loop, HF `generate()` | 48/48 identical |
| | chunked prefill vs single prefill | max difference 0.0 |
| `tests/test_engine.py` | 10 requests in 3 waves; engine output vs each request run alone | **10/10 identical** in all 4 configs |
| | ... including a paged cache too small for everyone (3 preemptions) | 10/10 identical |
| `tests/test_flash.py` | gpu-kernels FlashAttention vs PyTorch SDPA, kernel alone | max difference 1.2e-3 (T4, fp16) |
| | ... inside the whole model | 99.5% same top token at every position (T4, fp16) |

On the T4 in fp16, `test_vs_hf.py` also passes with 32/32 identical tokens, with and without the
gpu-kernels FlashAttention. `test_engine.py` in fp16 gave 10/10 identical requests in three of the
four configurations, and 9/10 in the one that forces preemption: the preempted request was
recomputed through a prefill instead of decode steps, which adds the same numbers in a different
order, and in fp16 that flipped a near-tie between two tokens later in the sequence. In fp32 all
four configurations are 10/10 identical, so the engine logic is exact; the fp16 test now requires
the first 8 tokens of each request to match.

Two things the tests caught:
- Qwen's instruct checkpoint ships `repetition_penalty: 1.1` in its generation config, which
  Hugging Face applies even in greedy mode. The comparison turns it off so both sides do plain
  greedy decoding.
- Exact token agreement is only a fair test in fp32, for the near-tie reason above.

## How it works

**KV cache (step 2).** To generate token *t*, attention needs the keys and values of every earlier
token. Without a cache, each step re-runs all 24 layers over the whole sequence. With one:

- **Prefill:** one pass over the prompt computes and stores K and V for every prompt token.
  This is compute-heavy and parallel, and it sets the time to first token.
- **Decode:** each step feeds only the newest token, appends its K and V, and attends over
  everything stored. Each step reads every weight to produce one token per sequence, so
  decode is **memory-bandwidth-bound**, the same regime as the softmax and LayerNorm kernels
  in gpu-kernels.

Memory per token: `24 layers × 2 (K and V) × 2 KV heads × 64 dims × 2 bytes (fp16) = 12 KB`.
Grouped-query attention keeps this small: with 14 KV heads it would be 7× larger.

**Attention contexts.** The model doesn't know about caches or batching. Each attention layer
computes q, k, v for the tokens it was given and asks a context object
([`attention.py`](tinyserve/attention.py)) for the keys and values to attend over and the
visibility mask. Prefill, single-sequence caching and batched decode are three small contexts,
so the model code is the same in every step.

**Continuous batching (step 4).** [`engine.py`](tinyserve/engine.py) re-decides the batch every
step: admit waiting requests while there is room (running each one's prefill), run one batched
decode pass where every request feeds its newest token at its *own* position, then retire
finished requests immediately. RoPE takes a `[batch, tokens]` position tensor, and a per-row
mask stops each request from seeing positions beyond its own length.

**Paged KV cache (step 5).** [`kv_cache.py`](tinyserve/kv_cache.py) keeps one pool of 16-token
blocks. Each sequence has a block table; a new block is taken only when a sequence crosses a
block boundary. If none is free, the engine preempts the most recently admitted request and
later re-prefills it from its prompt plus what it already generated (vLLM's recompute policy).
For attention, the blocks are gathered into a contiguous view and passed to PyTorch SDPA; vLLM
instead reads blocks directly inside its attention kernel, which avoids that copy.

**FlashAttention for prefill (step 3).** [`flash.py`](tinyserve/flash.py) compiles the
FlashAttention v3 kernel from gpu-kernels (PTX `mma.sync`, output accumulator in registers) as a
PyTorch extension. Prompts are zero-padded to a multiple of 64; padded keys come after every real
query, so the causal mask already hides them. Grouped-query K/V are expanded to 14 heads first.
Decode steps keep using SDPA, because this is a prefill kernel.

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m tests.test_vs_hf          # model vs Hugging Face
.venv/bin/python -m tests.test_engine         # batching and paging vs single requests
.venv/bin/python -m bench.bench_generate --skip-no-cache
.venv/bin/python -m bench.bench_load --requests 24 --min-prompt 16 --max-prompt 192 \
    --min-output 8 --max-output 64 --kv-budget 2048 --max-batch 32
.venv/bin/python -m tinyserve.chat "What is a KV cache?"
```

The first run downloads the model (~1 GB) from the Hugging Face Hub.

### Run on a GPU (e.g. Colab T4)

```bash
pip install -r requirements.txt
python -m tests.test_vs_hf --device cuda --dtype float16
python -m tests.test_vs_hf --device cuda --dtype float16 --attn-backend tinyflash
python -m tests.test_flash                         # step 3: kernel accuracy + prefill timing
python -m tests.test_engine --device cuda --dtype float16
python -m bench.bench_generate --prompt-lens 128 512 1024 2048 --new-tokens 128 --with-tinyflash
python -m bench.bench_load                         # 128 requests, 16k-token KV budget
```

`tests/test_flash.py` builds the CUDA extension on first use (about a minute). In fp16 the
tests require the first 8 tokens of each sequence to match rather than all of them, because a
near-tie between two tokens can flip when the same sums are added in a different order. The
strict, every-token checks are the CPU fp32 runs.

To run it in Colab with no setup: open a notebook with a T4 runtime, then
`!git clone https://github.com/Brinda-s/tinyserve`, `%cd tinyserve`,
`!pip install -q -r requirements.txt ninja`, and the commands above prefixed with `!`.

## What I'd do next

- **CUDA graphs for decode.** Single-request decode reaches only ~13% of the T4's memory bandwidth
  because of launch overhead. Capturing a decode step as a graph replays it with one launch.
- **Batched and chunked prefill.** New requests are prefilled one at a time, which pauses decoding
  for everyone else. Splitting long prompts into chunks and mixing them into decode steps evens
  out latency.
- **A paged-attention kernel.** The paged cache gathers each sequence's blocks into a contiguous
  copy before attention. Reading blocks directly inside the kernel, as vLLM does, removes that copy.

## Layout

```
tinyserve/model.py       Qwen2 decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU MLP
tinyserve/attention.py   attention contexts: prefill, single-sequence cache, batched decode
tinyserve/kv_cache.py    KVCache (step 2), SlotCache (step 4), PagedCache (step 5)
tinyserve/engine.py      continuous-batching scheduler with preemption
tinyserve/generate.py    single-request generation, with and without a cache
tinyserve/flash.py       gpu-kernels FlashAttention as a PyTorch extension (step 3)
tinyserve/csrc/          the CUDA binding and the kernel headers copied from gpu-kernels
tinyserve/loader.py      download from the Hub and load safetensors weights
tinyserve/chat.py        one-shot command-line chat
tests/                   correctness: vs Hugging Face, engine vs single requests, kernel vs SDPA
bench/                   single-request speed and the multi-request load test
```
