"""Step 4: a continuous-batching engine (with step 5's paged cache as one of its backends).

Static batching starts a batch of requests together and waits for the *longest* one to finish
before starting the next batch, so short requests sit idle and GPU slots go unused.
Continuous batching (iteration-level scheduling, from the Orca paper) re-decides the batch
on every step:

    each step:
      1. admit waiting requests while there is a batch slot and KV-cache room;
         run each one's prefill and sample its first token
      2. one batched decode forward pass: every running request feeds its newest token,
         each at its own position
      3. sample a token per request; requests that hit EOS or their length limit leave
         immediately and free their cache space for the next admission

With the paged cache, a running request may need a new block when none are free. The engine
then preempts the most recently admitted request: frees its blocks and puts it back at the
front of the queue. When re-admitted it is re-prefilled from its prompt plus the tokens it
already generated (vLLM's "recompute" preemption), so its output is unchanged.
"""
from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from .attention import AttnContext, PrefillContext
from .kv_cache import DecodeIndex, PagedCache, SlotCache
from .model import Qwen2

_ids = itertools.count()


@dataclass
class Request:
    prompt: list[int]
    max_new_tokens: int
    eos_ids: tuple[int, ...] = ()
    arrival: float = 0.0                       # seconds after the run starts
    id: int = field(default_factory=lambda: next(_ids))
    output: list[int] = field(default_factory=list)
    first_token_at: float | None = None
    finished_at: float | None = None
    preemptions: int = 0

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.output

    @property
    def done(self) -> bool:
        return len(self.output) >= self.max_new_tokens or (bool(self.output) and self.output[-1] in self.eos_ids)


class BatchDecodeContext(AttnContext):
    """Many sequences, one new token each: write it into the cache, read back each
    sequence's keys, and mask out positions beyond each sequence's own length."""

    def __init__(self, cache: SlotCache | PagedCache, idx: DecodeIndex):
        self.cache, self.idx = cache, idx
        keys = torch.arange(idx.span, device=idx.positions.device)
        self.mask = (keys[None, :] <= idx.positions[:, None])[:, None, None, :]  # [B, 1, 1, S]

    def kv(self, layer, k, v):
        self.cache.decode_write(layer, self.idx, k, v)
        return self.cache.decode_read(layer, self.idx)


@dataclass
class EngineStats:
    steps: int = 0
    decode_steps: int = 0
    decode_batch_total: int = 0
    prefills: int = 0
    preemptions: int = 0
    peak_running: int = 0
    peak_reserved_tokens: int = 0
    peak_used_tokens: int = 0

    @property
    def mean_decode_batch(self) -> float:
        return self.decode_batch_total / self.decode_steps if self.decode_steps else 0.0


class Engine:
    def __init__(self, model: Qwen2, cache: SlotCache | PagedCache, max_batch: int = 32,
                 policy: str = "continuous"):
        assert policy in ("continuous", "static")
        self.model, self.cache, self.max_batch, self.policy = model, cache, max_batch, policy
        self.device = next(model.parameters()).device
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.stats = EngineStats()
        self._t0 = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self._t0

    def add(self, req: Request) -> None:
        total = len(req.prompt) + req.max_new_tokens
        if total > self.cache.max_seq_tokens:
            raise ValueError(f"request {req.id} needs {total} tokens; the cache can hold at most "
                             f"{self.cache.max_seq_tokens} for one sequence")
        self.waiting.append(req)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------------ helpers
    def _finish(self, req: Request) -> None:
        req.finished_at = self.now()
        self.cache.release(req.id)
        self.finished.append(req)

    def _preempt(self, req: Request) -> None:
        self.cache.release(req.id)
        self.running.remove(req)
        self.waiting.appendleft(req)
        req.preemptions += 1
        self.stats.preemptions += 1

    @torch.inference_mode()
    def _prefill(self, req: Request) -> None:
        toks = req.tokens  # prompt, plus earlier output if this request was preempted
        ids = torch.tensor([toks], device=self.device)
        ctx = PrefillContext(store=lambda layer, k, v: self.cache.prefill_store(req.id, layer, k, v))
        logits = self.model(ids, ctx=ctx, last_only=True)[0, -1]
        req.output.append(int(logits.argmax(-1)))
        if req.first_token_at is None:
            req.first_token_at = self.now()
        self.stats.prefills += 1

    # --------------------------------------------------------------------- step
    @torch.inference_mode()
    def step(self) -> None:
        self.stats.steps += 1

        # 1. admission
        may_admit = self.policy == "continuous" or not self.running
        while may_admit and self.waiting and len(self.running) < self.max_batch:
            req = self.waiting[0]
            if not self.cache.can_admit(len(req.tokens), len(req.prompt) + req.max_new_tokens):
                break
            self.waiting.popleft()
            self.cache.admit(req.id, len(req.tokens))
            self._prefill(req)
            if req.done:
                self._finish(req)
            else:
                self.running.append(req)

        if not self.running:
            return

        # 2. make sure every running request has cache space for its newest token
        for req in list(self.running):
            if req not in self.running:
                continue  # preempted earlier in this loop
            while not self.cache.ensure(req.id, len(req.tokens)):
                victim = self.running[-1]
                self._preempt(victim)
                if victim is req:
                    break

        batch = list(self.running)
        if not batch:
            return
        self.stats.peak_running = max(self.stats.peak_running, len(batch))
        self.stats.peak_reserved_tokens = max(self.stats.peak_reserved_tokens, self.cache.reserved_tokens)
        self.stats.peak_used_tokens = max(self.stats.peak_used_tokens, self.cache.used_tokens)

        # 3. one batched decode step: each request feeds its newest token at its own position
        ids = torch.tensor([[r.tokens[-1]] for r in batch], device=self.device)
        positions = torch.tensor([len(r.tokens) - 1 for r in batch], device=self.device)
        idx = self.cache.decode_index([r.id for r in batch], positions)
        logits = self.model(ids, positions=positions[:, None], ctx=BatchDecodeContext(self.cache, idx),
                            last_only=True)[:, -1]
        next_tokens = logits.argmax(-1).tolist()
        self.stats.decode_steps += 1
        self.stats.decode_batch_total += len(batch)

        for req, tok in zip(batch, next_tokens):
            req.output.append(tok)
            if req.done:
                self.running.remove(req)
                self._finish(req)

    def run(self, requests: list[Request] | None = None) -> list[Request]:
        """Serve requests, honouring their arrival times, until all are finished."""
        pending = sorted(requests or [], key=lambda r: r.arrival)
        self._t0 = time.perf_counter()
        i = 0
        while i < len(pending) or self.has_work:
            while i < len(pending) and pending[i].arrival <= self.now():
                self.add(pending[i])
                i += 1
            if not self.has_work:
                time.sleep(max(0.0, pending[i].arrival - self.now()))
                continue
            self.step()
        return sorted(self.finished, key=lambda r: r.id)
