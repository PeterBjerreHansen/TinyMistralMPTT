#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tiny_mistral.device import resolve_device, resolve_dtype
from tiny_mistral.loading import load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--model-dir", default="checkpoints/TinyMistral-248M-v3")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--backend", choices=["auto", "reference", "flex", "local"], default="auto")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=None)
    args = p.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit('install tokenizer support with: pip install -e ".[io]"') from exc

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = Tokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))
    ids = tokenizer.encode(args.prompt).ids
    model = load_model(
        args.model_dir,
        attention_backend=args.backend,
        device=device,
        dtype=dtype,
    ).eval()
    if not ids or ids[0] != model.config.bos_token_id:
        ids = [model.config.bos_token_id] + ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    print(f"device={device} dtype={next(model.parameters()).dtype} backend={args.backend}")
    result = model.generate(
        input_ids,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
