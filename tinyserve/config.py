"""Model hyper-parameters, read from the Hugging Face config.json of a Qwen2-family model."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int           # d_model: 896 for Qwen2.5-0.5B
    intermediate_size: int     # MLP width: 4864
    num_layers: int            # 24
    num_heads: int             # query heads: 14
    num_kv_heads: int          # key/value heads: 2 (grouped-query attention, 7 query heads share one KV head)
    head_dim: int              # 64 -- same as gpu-kernels' FlashAttention
    rope_theta: float          # RoPE base frequency: 1,000,000
    rms_norm_eps: float
    tie_word_embeddings: bool  # 0.5B reuses the embedding matrix as the output layer
    max_position_embeddings: int

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        c = json.loads(Path(path).read_text())
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=c["num_attention_heads"],
            num_kv_heads=c["num_key_value_heads"],
            head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
            rope_theta=c.get("rope_theta", 10000.0),
            rms_norm_eps=c["rms_norm_eps"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
            max_position_embeddings=c["max_position_embeddings"],
        )
