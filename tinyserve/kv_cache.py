"""Key/value caches.

When generating token t, attention needs the keys and values of tokens 0..t-1. Those never
change once computed, so we store them instead of recomputing the whole prefix every step.

Three layouts, one per roadmap step:

  KVCache     (step 2) one sequence, one contiguous buffer sized for its maximum length.
  SlotCache   (step 4) a fixed number of slots for concurrent sequences; each slot reserves
              max_len tokens up front, whether the sequence ends up using them or not.
  PagedCache  (step 5) one shared pool of fixed-size blocks (16 tokens by default). A sequence
              gets blocks only as it grows, through a block table, like vLLM's PagedAttention.
              Memory is reserved for tokens that exist, so many more sequences fit.

SlotCache and PagedCache share one interface, used by the engine:
  can_admit(num_tokens, max_total)   is there room to start a sequence?
  admit(seq_id, num_tokens)          reserve space for its first num_tokens positions
  ensure(seq_id, length) -> bool     make sure positions 0..length-1 have space (False = out of memory)
  release(seq_id)                    give the space back
  prefill_store(seq_id, layer, k, v) write positions 0..T-1 of one sequence
  decode_index(seq_ids, positions)   precompute gather/scatter indices for one batched decode step
  decode_write / decode_read         per-layer write of the new token and read of all keys/values
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ModelConfig


def kv_bytes_per_token(cfg: ModelConfig, dtype: torch.dtype) -> int:
    """Memory for one token's keys and values across all layers."""
    return cfg.num_layers * 2 * cfg.num_kv_heads * cfg.head_dim * torch.tensor([], dtype=dtype).element_size()


class KVCache:
    """Step 2: a single sequence, one contiguous buffer per layer."""

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


@dataclass
class DecodeIndex:
    """Indices for one batched decode step, computed once and reused by all 24 layers."""
    rows: torch.Tensor        # [B] slot ids (SlotCache) or block ids of each new token (PagedCache)
    offsets: torch.Tensor     # [B] position within the slot / within the block
    positions: torch.Tensor   # [B] absolute position of each new token
    span: int                 # number of key positions to read: max(position) + 1
    tables: torch.Tensor | None = None  # PagedCache: [B, blocks] block table, padded


class SlotCache:
    """Step 4: max_seqs slots of max_len tokens each."""

    def __init__(self, cfg: ModelConfig, max_seqs: int, max_len: int, device, dtype):
        shape = (cfg.num_layers, max_seqs, cfg.num_kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len, self.device = max_len, device
        self._free = list(range(max_seqs - 1, -1, -1))
        self._slot: dict[int, int] = {}
        self._len: dict[int, int] = {}
        self.bytes_per_token = kv_bytes_per_token(cfg, dtype)

    @property
    def capacity_tokens(self) -> int:
        return self.k.shape[1] * self.max_len

    @property
    def max_seq_tokens(self) -> int:
        return self.max_len

    @property
    def reserved_tokens(self) -> int:
        return len(self._slot) * self.max_len

    @property
    def used_tokens(self) -> int:
        return sum(self._len.values())

    def can_admit(self, num_tokens: int, max_total: int) -> bool:
        return bool(self._free) and max_total <= self.max_len

    def admit(self, seq_id: int, num_tokens: int) -> None:
        self._slot[seq_id] = self._free.pop()
        self._len[seq_id] = num_tokens

    def ensure(self, seq_id: int, length: int) -> bool:
        if length > self.max_len:
            return False
        self._len[seq_id] = length
        return True

    def release(self, seq_id: int) -> None:
        self._free.append(self._slot.pop(seq_id))
        self._len.pop(seq_id)

    def prefill_store(self, seq_id: int, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        s, T = self._slot[seq_id], k.shape[2]
        self.k[layer, s, :, :T] = k[0]
        self.v[layer, s, :, :T] = v[0]

    def decode_index(self, seq_ids: list[int], positions: torch.Tensor) -> DecodeIndex:
        rows = torch.tensor([self._slot[i] for i in seq_ids], device=self.device)
        return DecodeIndex(rows, positions, positions, int(positions.max()) + 1)

    def decode_write(self, layer: int, idx: DecodeIndex, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k[layer][idx.rows, :, idx.offsets] = k[:, :, 0]
        self.v[layer][idx.rows, :, idx.offsets] = v[:, :, 0]

    def decode_read(self, layer: int, idx: DecodeIndex):
        return self.k[layer][idx.rows, :, :idx.span], self.v[layer][idx.rows, :, :idx.span]


class PagedCache:
    """Step 5: a shared pool of fixed-size blocks, allocated on demand through block tables."""

    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, device, dtype):
        shape = (cfg.num_layers, num_blocks, cfg.num_kv_heads, block_size, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.block_size, self.device = block_size, device
        self._free = list(range(num_blocks - 1, -1, -1))
        self._table: dict[int, list[int]] = {}
        self._len: dict[int, int] = {}
        self.bytes_per_token = kv_bytes_per_token(cfg, dtype)

    def _blocks_for(self, length: int) -> int:
        return -(-length // self.block_size)

    @property
    def capacity_tokens(self) -> int:
        return self.k.shape[1] * self.block_size

    @property
    def max_seq_tokens(self) -> int:
        # one block of headroom is kept free at admission (see can_admit)
        return (self.k.shape[1] - 1) * self.block_size

    @property
    def reserved_tokens(self) -> int:
        return sum(len(t) for t in self._table.values()) * self.block_size

    @property
    def used_tokens(self) -> int:
        return sum(self._len.values())

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    def can_admit(self, num_tokens: int, max_total: int) -> bool:
        # Room for the prompt plus one more block of growth, so a new sequence isn't
        # preempted on its very first decode step.
        return self._blocks_for(num_tokens) + 1 <= len(self._free)

    def admit(self, seq_id: int, num_tokens: int) -> None:
        self._table[seq_id] = []
        if not self.ensure(seq_id, num_tokens):
            raise RuntimeError("admit() called without enough free blocks")

    def ensure(self, seq_id: int, length: int) -> bool:
        table = self._table[seq_id]
        need = self._blocks_for(length) - len(table)
        if need > len(self._free):
            return False
        for _ in range(need):
            table.append(self._free.pop())
        self._len[seq_id] = length
        return True

    def release(self, seq_id: int) -> None:
        self._free.extend(reversed(self._table.pop(seq_id)))
        self._len.pop(seq_id, None)

    def prefill_store(self, seq_id: int, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        T = k.shape[2]
        pos = torch.arange(T, device=self.device)
        table = torch.tensor(self._table[seq_id], device=self.device)
        blocks, offs = table[pos // self.block_size], pos % self.block_size
        self.k[layer][blocks, :, offs] = k[0].transpose(0, 1)  # [T, KVH, D]
        self.v[layer][blocks, :, offs] = v[0].transpose(0, 1)

    def decode_index(self, seq_ids: list[int], positions: torch.Tensor) -> DecodeIndex:
        span = int(positions.max()) + 1
        nb = self._blocks_for(span)
        padded = [self._table[i][:nb] + [0] * (nb - min(nb, len(self._table[i]))) for i in seq_ids]
        tables = torch.tensor(padded, device=self.device)  # unused entries point at block 0; masked out
        rows = tables[torch.arange(len(seq_ids), device=self.device), positions // self.block_size]
        return DecodeIndex(rows, positions % self.block_size, positions, span, tables)

    def decode_write(self, layer: int, idx: DecodeIndex, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k[layer][idx.rows, :, idx.offsets] = k[:, :, 0]
        self.v[layer][idx.rows, :, idx.offsets] = v[:, :, 0]

    def decode_read(self, layer: int, idx: DecodeIndex):
        # Gather each sequence's blocks into a contiguous [B, KVH, blocks*block_size, D] view.
        # A production system (vLLM) reads the blocks inside the attention kernel instead of
        # gathering them first; this keeps the bookkeeping identical while using stock SDPA.
        B, nb = idx.tables.shape

        def gather(pool: torch.Tensor) -> torch.Tensor:
            g = pool[layer][idx.tables]                       # [B, nb, KVH, bs, D]
            g = g.permute(0, 2, 1, 3, 4).reshape(B, g.shape[2], nb * self.block_size, g.shape[4])
            return g[:, :, :idx.span]

        return gather(self.k), gather(self.v)
