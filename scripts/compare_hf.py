#!/usr/bin/env python
"""Compare full-model logits with Transformers 4.45.2 eager Mistral.

Run in a Python 3.13 environment with `pip install -e ".[dev]"`. To keep peak
memory modest, this computes and releases the HF model before loading the local
implementation.
"""
from __future__ import annotations

import argparse
import gc

import torch

from tiny_mistral.loading import load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", nargs="?", default="checkpoints/TinyMistral-248M-v3")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = p.parse_args()

    import transformers
    from transformers import AutoModelForCausalLM

    if transformers.__version__ != "4.45.2":
        raise RuntimeError(f"expected transformers==4.45.2, got {transformers.__version__}")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device(args.device)
    ids = torch.tensor([[1, 42, 314, 2718, 7, 99, 1234, 2]], dtype=torch.long, device=device)

    hf = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()
    with torch.no_grad():
        hf_logits = hf(input_ids=ids, use_cache=False).logits[:, :, :128].float().cpu()
    del hf
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ours = load_model(
        args.model_dir,
        attention_backend="reference",
        device=device,
        dtype=dtype,
        compile_flex=False,
    ).eval()
    with torch.no_grad():
        our_logits = ours(ids, use_cache=False).logits[:, :, :128].float().cpu()

    diff = (hf_logits - our_logits).abs()
    print(f"max_abs_diff={diff.max().item():.8g}")
    print(f"mean_abs_diff={diff.mean().item():.8g}")
    atol, rtol = ((2e-5, 2e-5) if dtype == torch.float32 else (8e-3, 8e-3))
    torch.testing.assert_close(our_logits, hf_logits, atol=atol, rtol=rtol)
    print("PASS: local reference logits match Transformers 4.45.2 eager logits")


if __name__ == "__main__":
    main()
