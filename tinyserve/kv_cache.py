"""Key/value cache for autoregressive decoding.

When generating token t, attention needs the keys and values of tokens 0..t-1.
Those never change once computed, so we store them instead of recomputing the
whole prefix every step. That turns each decode step from O(t) layer work
into O(1) layer work plus an O(t) attention read.

This is the simplest possible layout: one contiguous buffer per layer, sized
for the longest sequence up front. Step 5 of the roadmap replaces it with a
paged cache (fixed-size blocks handed out on demand), which is what lets a
server pack many requests of different lengths into the same memory.
"""
from __future__ import annotations

import torch

from .config import ModelConfig


class KVCache:
    def __init__(self, cfg: ModelConfig, batch_size: int, max_len: int, device, dtype):
        shape = (cfg.num_layers, batch_size, cfg.num_kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len

    def update(self, layer: int, start_pos: int, k: torch.Tensor, v: torch.Tensor):
        """Write k/v ([B, KVH, T, D]) at positions start_pos..start_pos+T-1 and return
        views of all keys/values for positions 0..start_pos+T-1."""
        end = start_pos + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache full: need {end} positions, have {self.max_len}")
        self.k[layer, :, :, start_pos:end] = k
        self.v[layer, :, :, start_pos:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2
