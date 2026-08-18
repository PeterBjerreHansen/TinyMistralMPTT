#!/usr/bin/env python
"""Compare real-checkpoint inputs_embeds forward and input-gradient parity."""
from __future__ import annotations

import argparse
import gc

import torch
import torch.nn.functional as F

from tiny_mistral.loading import load_model


def run_with_embeds(model, ids: torch.Tensor):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    output = model(inputs_embeds=embeds, use_cache=False, output_hidden_states=True)
    loss = F.cross_entropy(
        output.logits[:, :-1, :].reshape(-1, model.config.vocab_size),
        ids[:, 1:].reshape(-1),
    )
    embeds_grad = torch.autograd.grad(loss, embeds)[0]
    hidden = tuple(state.detach().float().cpu() for state in output.hidden_states or ())
    logits = output.logits.detach().float().cpu()
    return hidden, logits, embeds_grad.detach().float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default="checkpoints/TinyMistral-248M-v3")
    parser.add_argument("--length", type=int, default=40)
    args = parser.parse_args()
    import transformers
    from transformers import AutoModelForCausalLM
    if transformers.__version__ != "4.45.2":
        raise RuntimeError(f"expected transformers==4.45.2, got {transformers.__version__}")
    ids = ((torch.arange(args.length)[None, :] * 37 + 1) % 32005).long(); ids[:, 0] = 1
    hf = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    hf_hidden, hf_logits, hf_grad = run_with_embeds(hf, ids)
    del hf; gc.collect()
    ours = load_model(
        args.model_dir, attention_backend="reference", device="cpu", dtype=torch.float32,
        compile_flex=False,
    ).eval()
    our_hidden, our_logits, our_grad = run_with_embeds(ours, ids)
    if len(hf_hidden) != len(our_hidden):
        raise AssertionError("hidden-state tuple lengths differ")
    for index, (hf_state, our_state) in enumerate(zip(hf_hidden, our_hidden)):
        diff = (hf_state - our_state).abs()
        print(f"hidden[{index:02d}] max={diff.max().item():.8g} mean={diff.mean().item():.8g}")
        torch.testing.assert_close(our_state, hf_state, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(our_logits, hf_logits, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(our_grad, hf_grad, atol=2e-5, rtol=2e-5)
    print("PASS: inputs_embeds hidden states, logits, and gradients match Transformers 4.45.2")


if __name__ == "__main__":
    main()
