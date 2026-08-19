#!/usr/bin/env python
from __future__ import annotations

import argparse
from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a TinyMistral continued-pretraining experiment stage."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="exactly resume optimizer/RNG/data/pass-scheduler state",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="load model weights only and begin a fresh run",
    )
    parser.add_argument("--until-unique-tokens", type=int, default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if args.resume_from is not None and args.init_from is not None:
        parser.error("--resume-from and --init-from are mutually exclusive")
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        cfg.init_from = None
    elif args.init_from is not None:
        cfg.init_from = args.init_from
        cfg.resume_from = None
    cfg.validate()

    device = resolve_device(cfg.device)
    model = load_variant_from_config(cfg, device=device)
    train_data = PackedTokenDataset(cfg.data_dir, "train")
    validation_data = PackedTokenDataset(cfg.data_dir, "validation")

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
