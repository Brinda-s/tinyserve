"""Qwen2 / Qwen2.5 decoder written from scratch in plain PyTorch.

Architecture of one of the 24 layers (Qwen2.5-0.5B):

    x ─► RMSNorm ─► attention (GQA, RoPE, q/k/v with bias) ─► + ─► RMSNorm ─► SwiGLU MLP ─► + ─►
    └──────────────────────────────────────────────────────┘    └──────────────────────────┘

Module and parameter names deliberately match the Hugging Face checkpoint
(`layers.3.self_attn.q_proj.weight`, ...), so the safetensors file loads with
no renaming. The numerics also mirror Hugging Face (RMSNorm in fp32, RoPE
tables in fp32) so the correctness test can compare logits closely.

Positions are a [B, T] tensor, so one forward pass can carry several sequences that
are at different points of their generation (continuous batching). Where keys and
values come from is decided by an attention context (see attention.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .attention import AttnContext, PrefillContext, SingleCacheContext, run_attention
from .config import ModelConfig
from .kv_cache import KVCache


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight. No mean subtraction, no bias (unlike LayerNorm)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()  # accumulate in fp32, like Hugging Face, so fp16 runs stay stable
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding:
    """RoPE: rotates pairs of q/k features by an angle proportional to the token position,
    so q·k depends on the *relative* distance between tokens.

    Qwen uses the "rotate half" layout: feature i is paired with feature i + head_dim/2.
    The fp32 table grows on demand and is indexed by each token's own position.
    """

    def __init__(self, head_dim: int, theta: float):
        self.head_dim, self.theta = head_dim, theta
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def _build(self, length: int, device: torch.device) -> None:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [length, head_dim]
        self._cos, self._sin = emb.cos(), emb.sin()

    def __call__(self, positions: torch.Tensor, dtype: torch.dtype):
        """positions: [B, T] -> cos, sin of shape [B, 1, T, head_dim] (broadcast over heads)."""
        need = int(positions.max()) + 1
        if self._cos is None or self._cos.shape[0] < need or self._cos.device != positions.device:
            current = self._cos.shape[0] if self._cos is not None else 0
            self._build(max(need, 2 * current, 4096), positions.device)
        return self._cos[positions].unsqueeze(1).to(dtype), self._sin[positions].unsqueeze(1).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, layer_idx: int, ctx: AttnContext, backend: str):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)     # [B, H,   T, D]
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, KVH, T, D]
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # The context stores this step's k/v (if it has a cache) and returns every key/value
        # this step should attend over. Without a cache, each decode step would recompute
        # K and V for the whole prefix.
        k, v = ctx.kv(layer_idx, k, v)

        # Grouped-query attention: each KV head serves num_heads / num_kv_heads query heads.
        rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        out = run_attention(q, k, v, ctx, backend)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim))


class MLP(nn.Module):
    """SwiGLU: down( silu(gate(x)) * up(x) )."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, layer_idx, ctx, backend):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, layer_idx, ctx, backend)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.attn_backend = "sdpa"  # or "tinyflash": gpu-kernels' FlashAttention for causal prefill on CUDA

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, cache: KVCache | None = None,
                last_only: bool = False, *, positions: torch.Tensor | None = None,
                ctx: AttnContext | None = None) -> torch.Tensor:
        """input_ids: [B, T]. Returns logits [B, T, vocab], or [B, 1, vocab] with last_only.

        Simple use (one sequence): pass start_pos and optionally a KVCache.
        Engine use (many sequences): pass positions [B, T] and an attention context.
        """
        B, T = input_ids.shape
        if positions is None:
            positions = torch.arange(start_pos, start_pos + T, device=input_ids.device).expand(B, T)
        if ctx is None:
            ctx = PrefillContext() if cache is None else SingleCacheContext(cache, start_pos, T, input_ids.device)
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(positions, x.dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, i, ctx, self.attn_backend)
        x = self.norm(x)
        if last_only:
            x = x[:, -1:]
        return self.lm_head(x)

    @property
    def num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():  # tied embedding / lm_head counted once
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
