#!/usr/bin/env python
from __future__ import annotations

import argparse

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.model_factory import load_variant
from tiny_mistral_mptt.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vanilla continued-pretraining vertical slice.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    device = resolve_device(cfg.device)
    model = load_variant(
        cfg.variant,
        cfg.model_dir,
        device=device,
        dtype=cfg.dtype,
        attention_backend=cfg.attention_backend,
    )
    train_data = PackedTokenDataset(cfg.data_dir, "train")
    validation_data = PackedTokenDataset(cfg.data_dir, "validation")
    trainer = Trainer(
        model=model,
        config=cfg,
        train_data=train_data,
        validation_data=validation_data,
        device=device,
    )
    state = trainer.train()
    print(
        "PASS: training completed "
        f"steps={state.optimizer_steps} unique_tokens={state.unique_tokens_seen} "
        f"token_equivalent={state.token_equivalent_compute}"
    )


if __name__ == "__main__":
    main()
