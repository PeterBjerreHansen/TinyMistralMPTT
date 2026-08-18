#!/usr/bin/env python
"""Layer-by-layer diagnostic against Transformers 4.45.2 eager attention."""
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
    # Length 40 deliberately crosses TinyMistral's 32-position local window.
    ids = (torch.arange(40, dtype=torch.long)[None, :] * 37 + 1) % 32005
    ids[:, 0] = 1
    ids = ids.to(device)

    hf = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()
    with torch.no_grad():
        hf_out = hf(input_ids=ids, use_cache=False, output_hidden_states=True)
        hf_hidden = tuple(x[:, :, :64].float().cpu() for x in hf_out.hidden_states)
        hf_logits = hf_out.logits[:, :, :128].float().cpu()
    del hf, hf_out
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
        our_out = ours(ids, use_cache=False, output_hidden_states=True)
        our_hidden = tuple(x[:, :, :64].float().cpu() for x in our_out.hidden_states)
        our_logits = our_out.logits[:, :, :128].float().cpu()

    atol, rtol = ((2e-5, 2e-5) if dtype == torch.float32 else (8e-3, 8e-3))
    if len(hf_hidden) != len(our_hidden):
        raise AssertionError(f"hidden-state tuple lengths differ: HF={len(hf_hidden)} ours={len(our_hidden)}")
    for i, (a, b) in enumerate(zip(hf_hidden, our_hidden)):
        diff = (a - b).abs()
        print(f"hidden[{i:02d}] max={diff.max().item():.8g} mean={diff.mean().item():.8g}")
        torch.testing.assert_close(b, a, atol=atol, rtol=rtol)
    diff = (hf_logits - our_logits).abs()
    print(f"logits     max={diff.max().item():.8g} mean={diff.mean().item():.8g}")
    torch.testing.assert_close(our_logits, hf_logits, atol=atol, rtol=rtol)
    print("PASS: all hidden-state checkpoints and logits match the pinned HF oracle")


if __name__ == "__main__":
    main()
