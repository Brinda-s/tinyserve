"""Download a Qwen2-family checkpoint from the Hugging Face Hub and load it into our model."""
from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from .config import ModelConfig
from .model import Qwen2

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    # fp16 on GPUs (the T4 has no fast bf16); fp32 on CPU, where fp16 matmuls are slow.
    return torch.float16 if device.type in ("cuda", "mps") else torch.float32


def download(repo_id: str = DEFAULT_MODEL) -> Path:
    return Path(snapshot_download(repo_id, allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "*.txt"]))


def load_model(repo_id: str = DEFAULT_MODEL, device: torch.device | None = None,
               dtype: torch.dtype | None = None) -> tuple[Qwen2, Path]:
    device = device or pick_device()
    dtype = dtype or default_dtype(device)
    path = download(repo_id)
    cfg = ModelConfig.from_json(path / "config.json")

    # Build on the "meta" device (shapes only, no memory), then attach the real weights.
    with torch.device("meta"):
        model = Qwen2(cfg)

    state = {}
    for f in sorted(path.glob("*.safetensors")):
        for name, tensor in load_file(str(f)).items():
            state[name.removeprefix("model.")] = tensor.to(device=device, dtype=dtype)
    if cfg.tie_word_embeddings and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["embed_tokens.weight"]

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight  # share storage, as in the original model
    model.eval()
    return model, path
