#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.manifest import file_sha256
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.evaluation.pass_depth import evaluate_pass_depth
from tiny_mistral_mptt.model_factory import load_variant
from tiny_mistral_mptt.variants.multipass import MultiPassVariant


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NLL and hidden-state stability across recurrent pass depth.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--passes", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
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
    if not isinstance(model, MultiPassVariant):
        raise SystemExit("evaluate_pass_depth requires a multipass variant")
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        expected = file_sha256(f"{cfg.data_dir}/manifest.json")
        if payload.get("data_manifest_sha256") != expected:
            raise RuntimeError("checkpoint was trained against a different data manifest")
        model.load_state_dict(payload["model"], strict=True)
    dataset = PackedTokenDataset(cfg.data_dir, "validation")
    passes = cfg.eval_passes if args.passes is None else args.passes
    result = evaluate_pass_depth(
        model,
        dataset,
        device=device,
        passes=passes,
        max_blocks=args.max_blocks,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
