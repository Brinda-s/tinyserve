"""Step 6: load test. Same requests, same KV-cache memory budget, four ways of serving them.

    python -m bench.bench_load                     # GPU defaults: 128 requests
    python -m bench.bench_load --requests 24 --max-prompt 192 --max-output 64 --kv-budget 2048   # CPU-sized

  sequential            one request at a time (KV cache, no batching)
  static batching       start a batch, wait for all of it to finish, start the next
  continuous + slots    continuous batching; each request reserves max_len tokens of cache
  continuous + paged    continuous batching; cache handed out in 16-token blocks

Every configuration gets the same KV-cache budget in tokens, so the comparison shows what
each scheduling and memory strategy does with the same memory.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from tinyserve import DEFAULT_MODEL, load_model, pick_device
from tinyserve.engine import Engine, Request
from tinyserve.kv_cache import PagedCache, SlotCache
from tinyserve.loader import default_dtype


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--requests", type=int, default=128)
    ap.add_argument("--min-prompt", type=int, default=32)
    ap.add_argument("--max-prompt", type=int, default=512)
    ap.add_argument("--min-output", type=int, default=16)
    ap.add_argument("--max-output", type=int, default=256)
    ap.add_argument("--kv-budget", type=int, default=16384, help="KV cache size in tokens, for every config")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--rate", type=float, default=0.0, help="requests/s (Poisson); 0 = all arrive at once")
    ap.add_argument("--only", nargs="*", default=None, help="subset of: sequential static slots paged")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    dtype = default_dtype(device)
    model, _ = load_model(args.model, device, dtype)
    cfg = model.cfg

    rng = random.Random(0)
    max_len = args.max_prompt + args.max_output
    t, workload = 0.0, []
    for _ in range(args.requests):
        if args.rate > 0:
            t += rng.expovariate(args.rate)
        workload.append(([rng.randrange(1000, 30000) for _ in range(rng.randint(args.min_prompt, args.max_prompt))],
                         rng.randint(args.min_output, args.max_output), t))
    total_out = sum(n for _, n, _ in workload)
    slots = max(1, args.kv_budget // max_len)

    configs = {
        "sequential": (lambda: SlotCache(cfg, 1, max_len, device, dtype), "continuous", 1),
        "static": (lambda: SlotCache(cfg, slots, max_len, device, dtype), "static", slots),
        "slots": (lambda: SlotCache(cfg, slots, max_len, device, dtype), "continuous", slots),
        "paged": (lambda: PagedCache(cfg, args.kv_budget // args.block_size, args.block_size, device, dtype),
                  "continuous", args.max_batch),
    }
    labels = {"sequential": "sequential", "static": "static batching", "slots": "continuous + slots",
              "paged": "continuous + paged"}
    names = args.only or list(configs)

    dev_name = torch.cuda.get_device_name() if device.type == "cuda" else device.type
    print(f"{args.model} on {dev_name}, {str(dtype).removeprefix('torch.')}")
    print(f"{args.requests} requests, prompts {args.min_prompt}-{args.max_prompt}, outputs "
          f"{args.min_output}-{args.max_output} ({total_out} output tokens), "
          f"{'all at t=0' if args.rate <= 0 else f'{args.rate} req/s Poisson'}")
    print(f"KV budget {args.kv_budget} tokens = {args.kv_budget * model.cfg.num_layers * 2 * cfg.num_kv_heads * cfg.head_dim * (2 if dtype != torch.float32 else 4) / 2**20:.0f} MB "
          f"-> {slots} slots of {max_len} tokens, or {args.kv_budget // args.block_size} blocks of {args.block_size}\n")

    # warm-up (kernels, allocator)
    Engine(model, PagedCache(cfg, 64, 16, device, dtype)).run([Request([1, 2, 3, 4], 4)])

    rows = []
    header = (f"{'config':<20} {'tok/s':>8} {'wall s':>8} {'TTFT p50':>9} {'TTFT p95':>9} {'lat p50':>8} "
              f"{'lat p95':>8} {'batch':>6} {'peak':>5} {'mem used':>9} {'preempt':>8}")
    print(header)
    for name in names:
        make_cache, policy, max_batch = configs[name]
        eng = Engine(model, make_cache(), max_batch=max_batch, policy=policy)
        reqs = [Request(p, n, (), arrival=a) for p, n, a in workload]
        done = eng.run(reqs)
        wall = max(r.finished_at for r in done)
        ttft = [r.first_token_at - r.arrival for r in done]
        lat = [r.finished_at - r.arrival for r in done]
        s = eng.stats
        mem_eff = s.peak_used_tokens / s.peak_reserved_tokens if s.peak_reserved_tokens else 0.0
        row = {"config": labels[name], "throughput_tok_s": total_out / wall, "wall_s": wall,
               "ttft_p50_s": pct(ttft, 50), "ttft_p95_s": pct(ttft, 95), "latency_p50_s": pct(lat, 50),
               "latency_p95_s": pct(lat, 95), "mean_decode_batch": s.mean_decode_batch, "peak_running": s.peak_running,
               "kv_used_over_reserved": mem_eff, "preemptions": s.preemptions}
        rows.append(row)
        print(f"{labels[name]:<20} {row['throughput_tok_s']:>8.1f} {wall:>8.1f} {row['ttft_p50_s']:>8.2f}s "
              f"{row['ttft_p95_s']:>8.2f}s {row['latency_p50_s']:>7.1f}s {row['latency_p95_s']:>7.1f}s "
              f"{s.mean_decode_batch:>6.1f} {s.peak_running:>5} {mem_eff:>8.0%} {s.preemptions:>8}")

    print("\nbatch = mean decode batch size; peak = most requests running at once;")
    print("mem used = KV tokens actually filled / KV tokens reserved, at peak")
    out = Path("results"); out.mkdir(exist_ok=True)
    f = out / f"load_{device.type}.json"
    f.write_text(json.dumps({"device": dev_name, "args": vars(args), "results": rows}, indent=2))
    print(f"saved {f}")


if __name__ == "__main__":
    main()
