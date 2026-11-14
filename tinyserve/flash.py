"""Step 3: use the hand-written FlashAttention kernel from gpu-kernels for causal prefill.

The kernel (gpu-kernels FlashAttention v3) computes causal attention for fp16 inputs with
head dim 64 on sm_75+ GPUs. It is compiled the first time it is needed with PyTorch's
JIT extension builder (about a minute), then cached.

Two adaptations make it fit a real model:
  * Grouped-query attention: the model expands K/V to all 14 query heads before calling it.
  * Arbitrary prompt lengths: the kernel tiles in blocks of 64 rows, so the sequence is
    zero-padded to a multiple of 64. Padded keys come *after* every real query, so the
    causal mask already hides them; padded query rows are computed and then dropped.

Decode steps (one new token) still use PyTorch's SDPA: this kernel is a prefill kernel.
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

_CSRC = Path(__file__).parent / "csrc"


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load
    return load(name="tinyflash", sources=[str(_CSRC / "tinyflash.cu")],
                extra_include_paths=[str(_CSRC)], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


@lru_cache(maxsize=None)
def _device_ok(index: int) -> bool:
    if torch.cuda.get_device_capability(index) < (7, 5):  # mma.sync.m16n8k8 needs Turing or newer
        warnings.warn("tinyflash needs an sm_75+ GPU; falling back to PyTorch SDPA")
        return False
    try:
        _extension()
        return True
    except Exception as e:  # no compiler, wrong toolkit, ...
        warnings.warn(f"tinyflash failed to build ({e}); falling back to PyTorch SDPA")
        return False


def tinyflash_available(q: torch.Tensor) -> bool:
    return q.is_cuda and q.shape[-1] == 64 and _device_ok(q.device.index or 0)


def tinyflash_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q, k, v: [B, H, T, 64] (any float dtype) -> causal attention output [B, H, T, 64] in q's dtype."""
    B, H, T, D = q.shape
    pad = (-T) % 64

    def prep(x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float16)
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        return x.reshape(B * H, T + pad, D).contiguous()

    out = _extension().causal_attention(prep(q), prep(k), prep(v))
    return out.view(B, H, T + pad, D)[:, :, :T].to(q.dtype)
