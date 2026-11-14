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
| 2 | **KV cache**: prefill once, then decode one token at a time | ✅ 3.7–9× faster decode (CPU) |
| 3 | **Hand-written FlashAttention** from [gpu-kernels](https://github.com/Brinda-s/gpu-kernels) for prefill | ⏳ written; needs a CUDA GPU run |
| 4 | **Continuous batching**: requests join and leave the batch every step | ✅ token-identical to single requests |
| 5 | **Paged KV cache**: memory in 16-token blocks, with preemption when full | ✅ token-identical, incl. preemption |
| 6 | **Load test**: four serving strategies under the same memory budget | ✅ CPU results below; GPU next |

## Results

### Step 6: serving 24 requests with the same 48 MB of KV-cache memory (CPU, fp32)

24 requests arriving at once, prompts 16–192 tokens, outputs 8–64 tokens (749 output tokens total).
Every configuration gets a KV-cache budget of 2,048 tokens.

| Strategy | Throughput | Wall time | Median time to first token | Median latency | Mean batch | Peak concurrent | KV memory actually used |
|---|---|---|---|---|---|---|---|
| Sequential | 9.8 tok/s | 76.6 s | 38.7 s | 40.3 s | 1.0 | 1 | 82% |
| Static batching | 11.1 tok/s | 67.3 s | 11.7 s | 22.4 s | 4.1 | 8 | 49% |
| Continuous + slots | 13.0 tok/s | 57.5 s | 14.3 s | 35.5 s | 5.8 | 8 | 56% |
| **Continuous + paged** | **27.5 tok/s** | **27.2 s** | **6.3 s** | **20.4 s** | **9.0** | **18** | **95%** |

What the table shows:

- **Batching helps because decode is memory-bound.** Each decode step reads all ~2 GB of fp32
  weights whether it serves one request or nine, so more requests per step means more tokens
  per byte read.
- **Static batching wastes the batch.** A batch finishes only when its longest request does, so
  the mean batch size (4.1) stays far below the 8 slots available.
- **Continuous batching keeps slots full** by admitting a new request the moment one finishes.
- **Slots waste memory.** Each slot reserves room for the *longest possible* request (256 tokens),
  but the average request uses far less, so only 56% of reserved memory holds real tokens and
  only 8 requests fit.
- **Paged memory fixes that.** Blocks are handed out as sequences grow, 95% of reserved memory
  holds real tokens, and 18 requests run at once in the same 48 MB. That doubles throughput
  over slots. The 4 preemptions are the price: when blocks run out, the newest request is
  paused, its blocks freed, and it is recomputed later with identical output.

### Step 2: KV cache, single request (CPU, fp32, 64 new tokens)

| Prompt | Method | Time to first token | Decode tokens/s |
|---|---|---|---|
| 128 | no cache | 307 ms | 2.7 |
| 128 | **KV cache** | 197 ms | **10.1** (3.7×) |
| 128 | HF generate | 219 ms | 9.7 |
| 512 | no cache | 659 ms | 1.5 |
| 512 | **KV cache** | 601 ms | **13.9** (9×) |
| 512 | HF generate | 597 ms | 8.2 |

All CPU numbers are from an Apple-silicon laptop and are noisy (single runs). GPU numbers
(Colab T4, fp16) come next: see "Run on a GPU" below.

## Correctness

Every step is held to exact agreement, not "looks reasonable":

| Test | What it checks | Result (CPU, fp32) |
|---|---|---|
| `tests/test_vs_hf.py` | prompt logits vs the Hugging Face model | max difference **0.0** |
| | greedy tokens: our no-cache loop, our KV-cache loop, HF `generate()` | 48/48 identical |
| | chunked prefill vs single prefill | max difference 0.0 |
| `tests/test_engine.py` | 10 requests in 3 waves; engine output vs each request run alone | **10/10 identical** in all 4 configs |
| | ... including a paged cache too small for everyone (3 preemptions) | 10/10 identical |
| `tests/test_flash.py` | gpu-kernels FlashAttention vs PyTorch SDPA, kernel and whole model | needs a CUDA GPU |

One thing the tests caught: Qwen's instruct checkpoint ships `repetition_penalty: 1.1` in its
generation config, which Hugging Face applies even in greedy mode. The comparison turns it off
so both sides do plain greedy decoding.

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
greedy outputs can diverge from Hugging Face after a few tokens on near-ties, so the fp16 checks
in `test_vs_hf.py` require the first 8 tokens to match rather than all of them. `test_engine.py`
still demands exact matches, so in fp16 a batched run may differ from a single run on a rare
near-tie; the strict check is the CPU fp32 one.

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
