"""Step 3 check: gpu-kernels' FlashAttention against PyTorch SDPA (needs an sm_75+ CUDA GPU).

    python -m tests.test_flash

  1. kernel alone, random inputs, prompt lengths that are and are not multiples of 64
  2. whole model: prompt logits with attn_backend="tinyflash" vs "sdpa"
  3. prefill timing: tinyflash vs SDPA attention inside the model
"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIPPED: needs a CUDA GPU")
        return 0
    from tinyserve import load_model
    from tinyserve.flash import tinyflash_available, tinyflash_causal

    print(torch.cuda.get_device_name())
    probe = torch.zeros(1, 1, 1, 64, device="cuda")
    if not tinyflash_available(probe):
        print("FAIL: tinyflash not available on this GPU")
        return 1
    ok = True

    torch.manual_seed(0)
    for T in (1, 37, 64, 200, 512, 1031):
        q, k, v = (torch.randn(2, 14, T, 64, device="cuda", dtype=torch.float16) for _ in range(3))
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
        out = tinyflash_causal(q, k, v).float()
        err = (out - ref).abs().max().item()
        passed = err < 5e-3
        ok &= passed
        print(f"  [{'ok' if passed else 'FAIL':>4}] kernel vs SDPA, T={T:<5} max |diff| = {err:.2e}")

    model, _ = load_model(device=torch.device("cuda"), dtype=torch.float16)
    ids = torch.randint(0, model.cfg.vocab_size, (1, 777), device="cuda")
    with torch.inference_mode():
        model.attn_backend = "sdpa"
        a = model(ids).float()
        model.attn_backend = "tinyflash"
        b = model(ids).float()
    err = (a - b).abs().max().item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    passed = err < 0.5 and agree > 0.98
    ok &= passed
    print(f"  [{'ok' if passed else 'FAIL':>4}] model logits, tinyflash vs sdpa  max |diff| = {err:.2e}, "
          f"argmax agreement {agree:.1%}")

    print("\n  prefill time (whole model, fp16):")
    for T in (512, 1024, 2048, 4096):
        ids = torch.randint(0, model.cfg.vocab_size, (1, T), device="cuda")
        times = {}
        for backend in ("sdpa", "tinyflash"):
            model.attn_backend = backend
            with torch.inference_mode():
                model(ids, last_only=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(5):
                    model(ids, last_only=True)
                torch.cuda.synchronize()
            times[backend] = (time.perf_counter() - t0) / 5 * 1e3
        print(f"    T={T:<5} sdpa {times['sdpa']:7.1f} ms   tinyflash {times['tinyflash']:7.1f} ms   "
              f"({times['sdpa'] / times['tinyflash']:.2f}x)")

    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
