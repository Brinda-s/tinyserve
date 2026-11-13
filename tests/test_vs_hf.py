"""Check TinyServe against the official Hugging Face implementation of the same checkpoint.

    python -m tests.test_vs_hf                       # CPU, fp32: strict checks
    python -m tests.test_vs_hf --device cuda --dtype float16

Checks:
  1. prompt logits (every position) match Hugging Face
  2. greedy generation: our no-cache loop, our KV-cache loop and HF generate() pick the same tokens
  3. KV-cache logits match no-cache logits at every step (the cache changes speed, not results)
  4. chunked prefill (prompt fed in two pieces) matches a single prefill -- exercises the
     offset causal mask that continuous batching will rely on later
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinyserve import DEFAULT_MODEL, KVCache, generate, load_model

PROMPT = "Explain in a few sentences why a GPU is faster than a CPU for training neural networks."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--new-tokens", type=int, default=32)
    args = ap.parse_args()
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    strict = dtype == torch.float32
    logit_tol = 1e-2 if strict else 0.5  # logits are O(10); fp16 rounding moves them by ~0.1

    ours, path = load_model(args.model, device, dtype)
    tok = AutoTokenizer.from_pretrained(path)
    ref = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    eos = ref.generation_config.eos_token_id
    eos_ids = tuple(eos) if isinstance(eos, (list, tuple)) else (eos,)

    msgs = [{"role": "user", "content": PROMPT}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
    prompt_ids = list(prompt_ids)
    print(f"model={args.model} device={device} dtype={args.dtype} params={ours.num_params / 1e6:.0f}M "
          f"prompt={len(prompt_ids)} tokens")
    ok = True

    def report(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'ok' if passed else 'FAIL':>4}] {name:<36} {detail}")

    with torch.inference_mode():
        ids = torch.tensor([prompt_ids], device=device)
        a = ours(ids).float()
        b = ref(ids).logits.float()
        diff = (a - b).abs().max().item()
        agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        report("1. prompt logits vs HF", diff < logit_tol, f"max |diff| = {diff:.2e}, argmax agreement {agree:.1%}")

    nc = generate(ours, prompt_ids, args.new_tokens, use_cache=False, eos_ids=eos_ids, keep_logits=True)
    kv = generate(ours, prompt_ids, args.new_tokens, use_cache=True, eos_ids=eos_ids, keep_logits=True)
    with torch.inference_mode():
        # Plain greedy decoding. The instruct checkpoint's generation_config sets repetition_penalty=1.1,
        # which HF applies even with do_sample=False; turn it off so both sides do the same thing.
        hf_out = ref.generate(ids, max_new_tokens=args.new_tokens, do_sample=False, temperature=None, top_p=None,
                              top_k=None, repetition_penalty=1.0,
                              pad_token_id=tok.eos_token_id)[0, len(prompt_ids):].tolist()

    def same_prefix(x: list[int], y: list[int]) -> int:
        n = 0
        for p, q in zip(x, y):
            if p != q:
                break
            n += 1
        return n

    m1, m2 = same_prefix(kv.tokens, hf_out), same_prefix(kv.tokens, nc.tokens)
    need = len(hf_out) if strict else min(8, len(hf_out))  # fp16 may drift on near-ties later on
    report("2a. greedy tokens: cache vs HF", m1 >= need and (not strict or kv.tokens == hf_out),
           f"{m1}/{len(hf_out)} identical")
    report("2b. greedy tokens: cache vs no-cache", m2 >= need and (not strict or kv.tokens == nc.tokens),
           f"{m2}/{len(nc.tokens)} identical")

    n = min(len(kv.step_logits), len(nc.step_logits), m2 + 1)
    step_diff = max((x - y).abs().max().item() for x, y in zip(kv.step_logits[:n], nc.step_logits[:n]))
    report("3. per-step logits: cache vs no-cache", step_diff < logit_tol, f"max |diff| = {step_diff:.2e} over {n} steps")

    with torch.inference_mode():
        split = len(prompt_ids) // 2
        cache = KVCache(ours.cfg, 1, len(prompt_ids), device, dtype)
        ours(ids[:, :split], start_pos=0, cache=cache)
        chunked = ours(ids[:, split:], start_pos=split, cache=cache).float()
        single = a[:, split:]
        d = (chunked - single).abs().max().item()
        report("4. chunked prefill vs single prefill", d < logit_tol, f"max |diff| = {d:.2e}")

    print("\nTinyServe output:", repr(tok.decode(kv.tokens)))
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
