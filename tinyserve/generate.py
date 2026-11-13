"""Token generation: the naive loop (no cache) and the KV-cache loop (prefill + decode)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .kv_cache import KVCache
from .model import Qwen2


@dataclass
class GenerationResult:
    tokens: list[int]                       # newly generated tokens only
    ttft_s: float                           # time to first token: prompt processing + first sample
    total_s: float
    step_logits: list[torch.Tensor] = field(default_factory=list, repr=False)  # kept for tests

    @property
    def decode_tokens_per_s(self) -> float:
        n = len(self.tokens) - 1  # tokens after the first
        return n / (self.total_s - self.ttft_s) if n > 0 and self.total_s > self.ttft_s else float("nan")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _sample(logits: torch.Tensor, temperature: float, generator: torch.Generator | None) -> int:
    if temperature <= 0:
        return int(logits.argmax(-1))
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.inference_mode()
def generate(model: Qwen2, prompt_ids: list[int], max_new_tokens: int, *, use_cache: bool = True,
             eos_ids: tuple[int, ...] = (), temperature: float = 0.0, seed: int = 0,
             keep_logits: bool = False) -> GenerationResult:
    """Generate up to max_new_tokens tokens for a single prompt (batch size 1).

    use_cache=False: every step re-runs the model over the *whole* sequence (prompt + generated
                     so far) and keeps only the last position. Cost per step grows with length.
    use_cache=True:  one *prefill* pass over the prompt fills the KV cache; each *decode* step
                     then feeds just the newest token.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    gen = torch.Generator(device="cpu").manual_seed(seed) if temperature > 0 else None
    seq = list(prompt_ids)
    out: list[int] = []
    logits_log: list[torch.Tensor] = []
    cache = KVCache(model.cfg, 1, len(prompt_ids) + max_new_tokens, device, dtype) if use_cache else None

    _sync(device)
    t0 = time.perf_counter()
    ttft = 0.0
    for step in range(max_new_tokens):
        if not use_cache:
            ids = torch.tensor([seq], device=device)
            logits = model(ids, start_pos=0, cache=None, last_only=True)[0, -1]
        elif step == 0:  # prefill
            ids = torch.tensor([seq], device=device)
            logits = model(ids, start_pos=0, cache=cache, last_only=True)[0, -1]
        else:  # decode: only the newest token, at position len(seq) - 1
            ids = torch.tensor([[seq[-1]]], device=device)
            logits = model(ids, start_pos=len(seq) - 1, cache=cache, last_only=True)[0, -1]

        tok = _sample(logits, temperature, gen)
        if keep_logits:
            logits_log.append(logits.float().cpu())
        if step == 0:
            _sync(device)
            ttft = time.perf_counter() - t0
        seq.append(tok)
        out.append(tok)
        if tok in eos_ids:
            break
    _sync(device)
    return GenerationResult(out, ttft, time.perf_counter() - t0, logits_log)
