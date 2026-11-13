"""Qwen2 / Qwen2.5 decoder written from scratch in plain PyTorch.

Architecture of one of the 24 layers (Qwen2.5-0.5B):

    x ─► RMSNorm ─► attention (GQA, RoPE, q/k/v with bias) ─► + ─► RMSNorm ─► SwiGLU MLP ─► + ─►
    └──────────────────────────────────────────────────────┘    └──────────────────────────┘

Module and parameter names deliberately match the Hugging Face checkpoint
(`layers.3.self_attn.q_proj.weight`, ...), so the safetensors file loads with
no renaming. The numerics also mirror Hugging Face (RMSNorm in fp32, RoPE
tables in fp32) so the correctness test can compare logits closely.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

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
    Tables are built once in fp32 up to the longest position seen, then sliced per call.
    """

    def __init__(self, head_dim: int, theta: float):
        self.head_dim = head_dim
        self.theta = theta
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def _build(self, length: int, device: torch.device) -> None:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        pos = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)            # [length, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)       # [length, head_dim]
        self._cos, self._sin = emb.cos(), emb.sin()

    def __call__(self, start: int, length: int, device: torch.device, dtype: torch.dtype):
        end = start + length
        if self._cos is None or self._cos.shape[0] < end or self._cos.device != device:
            self._build(max(end, 2 * (self._cos.shape[0] if self._cos is not None else 0), 4096), device)
        return self._cos[start:end].to(dtype), self._sin[start:end].to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, heads, T, D]; cos/sin: [T, D] broadcast over batch and heads
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, layer_idx: int, start_pos: int, cache: KVCache | None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)     # [B, H,   T, D]
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, KVH, T, D]
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:
            # Store this step's keys/values, then attend over everything cached so far.
            # Without a cache, every decode step would recompute K and V for the whole prefix.
            k, v = cache.update(layer_idx, start_pos, k, v)

        # Grouped-query attention: each KV head serves num_heads / num_kv_heads query heads.
        rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        S = k.shape[2]  # number of keys attended to
        if T == 1:
            out = F.scaled_dot_product_attention(q, k, v)  # one new token sees every earlier token
        elif start_pos == 0 and S == T:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Chunk of T new tokens appended after start_pos cached ones: query i sees keys <= start_pos + i.
            q_pos = torch.arange(start_pos, start_pos + T, device=x.device)[:, None]
            k_pos = torch.arange(S, device=x.device)[None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=k_pos <= q_pos)

        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


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

    def forward(self, x, cos, sin, layer_idx, start_pos, cache):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, layer_idx, start_pos, cache)
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

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, cache: KVCache | None = None,
                last_only: bool = False) -> torch.Tensor:
        """input_ids: [B, T] tokens occupying positions start_pos .. start_pos+T-1.
        Returns logits [B, T, vocab], or [B, 1, vocab] with last_only (all generation needs)."""
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(start_pos, input_ids.shape[1], x.device, x.dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, i, start_pos, cache)
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


def flops_per_token(cfg: ModelConfig, context: int) -> float:
    """Rough FLOPs for one token: 2 per weight (matmuls) + attention over `context` keys."""
    d, h = cfg.hidden_size, cfg.num_heads * cfg.head_dim
    kv = cfg.num_kv_heads * cfg.head_dim
    per_layer = 2 * (d * h + 2 * d * kv + h * d + 3 * d * cfg.intermediate_size) + 4 * h * context
    return cfg.num_layers * per_layer + 2 * d * cfg.vocab_size

