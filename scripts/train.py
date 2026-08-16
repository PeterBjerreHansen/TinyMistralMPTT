#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from tiny_mistral_mptt.variants.fbt import FBTVariant
from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.model_factory import load_variant
from tiny_mistral_mptt.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TinyMistral continued-pretraining experiment stage.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None, help="exactly resume optimizer/RNG/data/pass-scheduler state")
    parser.add_argument("--init-from", default=None, help="load model weights only and begin a fresh run")
    parser.add_argument("--until-unique-tokens", type=int, default=None)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.init_from is not None:
        cfg.init_from = args.init_from
    cfg.validate()
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
    train_data = PackedTokenDataset(cfg.data_dir, "train")
    validation_data = PackedTokenDataset(cfg.data_dir, "validation")
    if cfg.fbt_initialization == "calibrated":
        if not isinstance(model, FBTVariant):
            raise SystemExit("calibrated FBT initialization requires variant=fbt")
        calibration_data = train_data if cfg.fbt_calibration_split == "train" else validation_data
        if not 0 <= cfg.fbt_calibration_block < len(calibration_data):
            raise SystemExit(
                f"fbt_calibration_block must be in [0, {len(calibration_data)})"
            )
        calibration_ids = calibration_data.batch([cfg.fbt_calibration_block], device=device)
        calibration_stats = model.calibrate_initialization(
            calibration_ids,
            gate_logit_std_target=cfg.fbt_gate_logit_std_target,
        )
        model.initialization_stats = calibration_stats
        print("FBT initialization calibration " + json.dumps(calibration_stats, sort_keys=True))
    trainer = Trainer(
        model=model,
        config=cfg,
        train_data=train_data,
        validation_data=validation_data,
        device=device,
    )
    state = trainer.train(until_unique_tokens=args.until_unique_tokens)
    print(
        "PASS: training completed "
        f"phase={state.phase} steps={state.optimizer_steps} "
        f"unique_tokens={state.unique_tokens_seen} "
        f"token_equivalent={state.token_equivalent_compute}"
    )


if __name__ == "__main__":
    main()
