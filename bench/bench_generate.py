"""Benchmark single-request generation: no cache vs KV cache vs Hugging Face generate().

    python -m bench.bench_generate                         # auto device (cuda > mps > cpu)
    python -m bench.bench_generate --prompt-lens 128 512 1024 --new-tokens 128

Reports, per prompt length:
  TTFT          time to first token (prefill)
  decode tok/s  tokens per second after the first one
Writes results/bench_<device>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinyserve import DEFAULT_MODEL, generate, load_model, pick_device
from tinyserve.loader import default_dtype

TEXT = ("GPUs hide memory latency by keeping many warps in flight. Large language models generate text one "
        "token at a time, and every step reads all of the model weights from memory. ")


def make_prompt(tok, n: int) -> list[int]:
    ids: list[int] = []
    while len(ids) < n:
        ids += tok(TEXT, add_special_tokens=False)["input_ids"]
    return ids[:n]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, choices=[None, "float32", "float16", "bfloat16"])
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-no-cache", action="store_true", help="the no-cache loop is slow on CPU")
    ap.add_argument("--with-tinyflash", action="store_true",
                    help="also run the KV-cache loop with gpu-kernels FlashAttention for prefill (CUDA)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    dtype = getattr(torch, args.dtype) if args.dtype else default_dtype(device)
    model, path = load_model(args.model, device, dtype)
    tok = AutoTokenizer.from_pretrained(path)
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    name = torch.cuda.get_device_name() if device.type == "cuda" else device.type
    print(f"{args.model} on {name}, {str(dtype).removeprefix('torch.')}, {model.num_params / 1e6:.0f}M params, "
          f"{args.new_tokens} new tokens, median of {args.repeats}\n")

    def ours(prompt, use_cache, backend="sdpa"):
        model.attn_backend = backend
        r = generate(model, prompt, args.new_tokens, use_cache=use_cache)  # no EOS: always full length
        return r.ttft_s, r.decode_tokens_per_s

    @torch.inference_mode()
    def hf_run(prompt):
        ids = torch.tensor([prompt], device=device)
        kw = dict(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                  pad_token_id=tok.eos_token_id)
        sync(device); t0 = time.perf_counter()
        hf.generate(ids, max_new_tokens=1, min_new_tokens=1, **kw)
        sync(device); ttft = time.perf_counter() - t0
        t0 = time.perf_counter()
        hf.generate(ids, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, **kw)
        sync(device); total = time.perf_counter() - t0
        return ttft, (args.new_tokens - 1) / (total - ttft)

    results = []
    print(f"{'prompt':>7}  {'method':<26} {'TTFT ms':>9} {'decode tok/s':>13}")
    for n in args.prompt_lens:
        prompt = make_prompt(tok, n)
        methods = [("tinyserve no cache", lambda: ours(prompt, False)),
                   ("tinyserve KV cache", lambda: ours(prompt, True)),
                   ("HF generate", lambda: hf_run(prompt))]
        if args.skip_no_cache:
            methods = methods[1:]
        if args.with_tinyflash:
            methods.insert(-1, ("tinyserve KV + tinyflash", lambda: ours(prompt, True, "tinyflash")))
        for label, fn in methods:
            fn()  # warm-up
            runs = [fn() for _ in range(args.repeats)]
            ttft = statistics.median(r[0] for r in runs) * 1e3
            tps = statistics.median(r[1] for r in runs)
            print(f"{n:>7}  {label:<26} {ttft:>9.1f} {tps:>13.1f}")
            results.append({"prompt_len": n, "method": label, "ttft_ms": ttft, "decode_tok_s": tps})
        print()

    out = Path("results"); out.mkdir(exist_ok=True)
    f = out / f"bench_{device.type}.json"
    f.write_text(json.dumps({"device": name, "dtype": str(dtype), "model": args.model,
                             "new_tokens": args.new_tokens, "results": results}, indent=2))
    print(f"saved {f}")


if __name__ == "__main__":
    main()
