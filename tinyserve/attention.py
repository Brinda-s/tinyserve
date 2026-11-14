"""Attention contexts: where a layer's keys/values come from and which keys each query may see.

The model itself knows nothing about caches or batching. Every attention layer computes
q, k, v for the tokens it was given and hands k, v to a context object, which returns the
full set of keys/values to attend over plus the visibility rule. Three contexts cover
every case in TinyServe:

  PrefillContext       all tokens are new and start at position 0 -> plain causal attention.
                       Optionally stores k/v into a cache as a side effect.
  SingleCacheContext   one sequence with a contiguous KVCache (the step-2 generate() loop).
  BatchDecodeContext   many sequences, one new token each, at different positions, reading
                       from a slot or paged cache (steps 4-5; built by the engine).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .kv_cache import KVCache


class AttnContext:
    causal_prefill: bool = False        # True: keys == the given tokens, causal mask, positions from 0
    mask: torch.Tensor | None = None    # bool [B, 1, T, S], True = may attend; None = attend to all

    def kv(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class PrefillContext(AttnContext):
    causal_prefill = True

    def __init__(self, store: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None):
        self.store = store

    def kv(self, layer, k, v):
        if self.store is not None:
            self.store(layer, k, v)
        return k, v


class SingleCacheContext(AttnContext):
    def __init__(self, cache: KVCache, start_pos: int, num_new: int, device: torch.device):
        self.cache, self.start_pos = cache, start_pos
        total = start_pos + num_new
        if num_new > 1 and start_pos > 0:
            # Chunk appended after cached tokens: new token i sees keys 0 .. start_pos + i.
            q_pos = torch.arange(start_pos, total, device=device)[:, None]
            self.mask = (torch.arange(total, device=device)[None, :] <= q_pos)[None, None]
        elif num_new > 1:
            self.mask = torch.ones(num_new, num_new, dtype=torch.bool, device=device).tril()[None, None]
        # num_new == 1: the newest token sees everything -> mask stays None

    def kv(self, layer, k, v):
        return self.cache.update(layer, self.start_pos, k, v)


def run_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx: AttnContext, backend: str) -> torch.Tensor:
    """q: [B, H, T, D]; k, v: [B, H, S, D] (already expanded for grouped-query attention)."""
    if ctx.causal_prefill:
        if backend == "tinyflash":
            from .flash import tinyflash_available, tinyflash_causal
            if tinyflash_available(q):
                return tinyflash_causal(q, k, v)
        return F.scaled_dot_product_attention(q, k, v, is_causal=q.shape[2] > 1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=ctx.mask)
