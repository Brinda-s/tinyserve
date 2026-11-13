"""Ask the model one question from the command line, using TinyServe's own generation loop.

    python -m tinyserve.chat "What is a KV cache?"
"""
from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from . import DEFAULT_MODEL, generate, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    model, path = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True,
                                  tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    eos = tuple(t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None)
    r = generate(model, list(ids), args.max_new_tokens, eos_ids=eos, temperature=args.temperature)
    print(tok.decode(r.tokens, skip_special_tokens=True))
    print(f"\n[{len(r.tokens)} tokens, first token after {r.ttft_s * 1e3:.0f} ms, "
          f"then {r.decode_tokens_per_s:.1f} tokens/s]")


if __name__ == "__main__":
    main()
