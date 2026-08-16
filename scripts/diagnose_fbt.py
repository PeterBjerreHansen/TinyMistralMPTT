#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.manifest import file_sha256
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.model_factory import load_variant
from tiny_mistral_mptt.variants.fbt import FBTVariant


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure FBT feedback and gate scales on a fixed validation block.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--block-index", type=int, default=0)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if cfg.variant != "fbt":
        raise SystemExit("diagnose_fbt requires variant=fbt")
    device = resolve_device(cfg.device)
    model = load_variant(
        cfg.variant,
        cfg.model_dir,
        device=device,
        dtype=cfg.dtype,
        attention_backend=cfg.attention_backend,
        architecture_seed=cfg.architecture_seed,
        memory_window=cfg.memory_window,
        prefix_mixin_probability=cfg.prefix_mixin_probability,
    )
    if not isinstance(model, FBTVariant):
        raise SystemExit("diagnose_fbt requires an FBT model")
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        expected = file_sha256(f"{cfg.data_dir}/manifest.json")
        if payload.get("data_manifest_sha256") != expected:
            raise RuntimeError("checkpoint was trained against a different data manifest")
        model.load_state_dict(payload["model"], strict=True)
    model.eval()
    dataset = PackedTokenDataset(cfg.data_dir, "validation")
    if not 0 <= args.block_index < len(dataset):
        raise SystemExit(f"block index must be in [0, {len(dataset)})")
    ids = dataset.batch([args.block_index], device=device)
    with torch.no_grad():
        token_embeddings = model.backbone.model.embed_tokens(ids)
        previous_hidden = model._run_first_hidden(ids)
        shifted = model.shift_previous(previous_hidden)
        feedback_value = model.feedback_value(shifted)
        gate_logits = model.feedback_gate(token_embeddings)
        gate = torch.sigmoid(gate_logits)
        fused = feedback_value * gate
        feedback_inputs = model.feedback_inputs(token_embeddings, previous_hidden)
        result = {
            "block_index": args.block_index,
            "prefix_mixin_probability": model.prefix_mixin_probability,
            "rms_token_embedding_e": _rms(token_embeddings),
            "rms_previous_top_hidden_h": _rms(previous_hidden),
            "rms_feedback_value_WUh": _rms(feedback_value),
            "gate_mean": float(gate.mean().detach().cpu()),
            "gate_std": float(gate.std().detach().cpu()),
            "gate_logit_mean": float(gate_logits.mean().detach().cpu()),
            "gate_logit_std": float(gate_logits.std().detach().cpu()),
            "rms_fused_nonzero_positions": _rms(fused[:, 1:, :]),
            "rms_actual_feedback_input": _rms(feedback_inputs[:, 1:, :]),
            "rms_position0_embedding": _rms(feedback_inputs[:, :1, :]),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
