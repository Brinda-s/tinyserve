"""Steps 4-5 check: batched serving must give exactly the tokens single-request generation gives.

    python -m tests.test_engine                       # CPU fp32
    python -m tests.test_engine --device cuda

Ten requests of different lengths arrive in three waves while others are still running.
Each engine configuration must reproduce, token for token, what tinyserve.generate()
produces for each request on its own (greedy decoding, EOS honoured):

  continuous batching + slot cache
  continuous batching + paged cache
  continuous batching + paged cache too small for everyone -> forces preemption
  static batching + paged cache
"""
from __future__ import annotations

import argparse
import random
import sys

import torch
from transformers import AutoTokenizer

from tinyserve import DEFAULT_MODEL, generate, load_model
from tinyserve.engine import Engine, Request
from tinyserve.kv_cache import PagedCache, SlotCache

QUESTIONS = [
    "What is 17 times 23?", "Name three uses of a GPU.", "Write a haiku about routers.",
    "Explain DNS in one sentence.", "What does a KV cache store?", "Give two tips for faster Wi-Fi.",
]


def build_requests(tok, eos_ids) -> list[tuple[list[int], int]]:
    rng = random.Random(0)
    specs = []
    for q in QUESTIONS:  # real chat prompts that may stop early at EOS
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)
        specs.append((ids, rng.randint(8, 40)))
    for n in (3, 17, 64, 97):  # raw token prompts of awkward lengths (block size is 16)
        specs.append(([rng.randrange(1000, 20000) for _ in range(n)], rng.randint(5, 30)))
    return specs


def run_engine(model, cache, specs, eos_ids, policy="continuous", max_batch=8):
    eng = Engine(model, cache, max_batch=max_batch, policy=policy)
    reqs = [Request(p, n, eos_ids) for p, n in specs]
    waves = [reqs[:4], reqs[4:7], reqs[7:]]  # later waves join while earlier requests are mid-generation
    for w, wave in enumerate(waves):
        for r in wave:
            eng.add(r)
        for _ in range(3 if w < len(waves) - 1 else 0):
            eng.step()
    while eng.has_work:
        eng.step()
    return reqs, eng.stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)

    model, path = load_model(args.model, device, dtype)
    tok = AutoTokenizer.from_pretrained(path)
    eos_ids = tuple(t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None)
    specs = build_requests(tok, eos_ids)
    expected = [generate(model, p, n, eos_ids=eos_ids).tokens for p, n in specs]
    longest = max(len(p) + n for p, n in specs)
    print(f"{len(specs)} requests, prompts {min(len(p) for p, _ in specs)}-{max(len(p) for p, _ in specs)} tokens, "
          f"longest sequence {longest} tokens")

    configs = [
        ("continuous + slot cache", lambda: SlotCache(model.cfg, 8, 256, device, dtype), "continuous"),
        ("continuous + paged cache", lambda: PagedCache(model.cfg, 128, 16, device, dtype), "continuous"),
        ("continuous + small paged cache", lambda: PagedCache(model.cfg, -(-longest // 16) + 3, 16, device, dtype),
         "continuous"),
        ("static + paged cache", lambda: PagedCache(model.cfg, 128, 16, device, dtype), "static"),
    ]
    # fp32: every request must be token-identical. fp16/bf16: a batched or re-prefilled request runs
    # the same math in a different order, so a near-tie between two tokens can flip late in a
    # sequence. There, each request must match for its first 8 tokens (or all of it, if shorter).
    strict = dtype == torch.float32

    def agree(x: list[int], y: list[int]) -> int:
        n = 0
        while n < min(len(x), len(y)) and x[n] == y[n]:
            n += 1
        return n

    ok = True
    for name, make_cache, policy in configs:
        reqs, stats = run_engine(model, make_cache(), specs, eos_ids, policy)
        same = sum(r.output == e for r, e in zip(reqs, expected))
        close = all(r.output == e or agree(r.output, e) >= min(8, len(e)) for r, e in zip(reqs, expected))
        passed = same == len(specs) if strict else close
        ok &= passed
        print(f"  [{'ok' if passed else 'FAIL':>4}] {name:<32} {same}/{len(specs)} requests identical, "
              f"mean decode batch {stats.mean_decode_batch:.1f}, preemptions {stats.preemptions}")
        for i, (r, e) in enumerate(zip(reqs, expected)):
            if r.output != e:
                print(f"         request {i}: same first {agree(r.output, e)} of {len(e)} tokens, then diverges")
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
