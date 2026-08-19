#!/usr/bin/env python
from __future__ import annotations

import argparse
import signal

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.model_factory import load_variant
from tiny_mistral_mptt.training.trainer import Trainer


_STOP_REQUESTED = False


def _request_stop(signum, frame) -> None:  # pragma: no cover - OS integration
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a TinyMistral continued-pretraining experiment stage."
    )
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume-from",
        default=None,
        help="exactly resume optimizer/RNG/data/pass-scheduler state",
    )
    group.add_argument(
        "--resume-auto",
        action="store_true",
        help="start a new run if output_dir is empty, otherwise resume the newest valid generation",
    )
    group.add_argument(
        "--init-from",
        default=None,
        help="load model weights only and begin a fresh run",
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="development-only escape hatch for resuming with a different Git/uv.lock identity",
    )
    parser.add_argument("--until-unique-tokens", type=int, default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        cfg.init_from = None
    elif args.init_from is not None:
        cfg.init_from = args.init_from
        cfg.resume_from = None
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

    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)

    trainer = Trainer(
        model=model,
        config=cfg,
        train_data=train_data,
        validation_data=validation_data,
        device=device,
        resume_auto=args.resume_auto,
        allow_source_mismatch=args.allow_source_mismatch,
        stop_requested=lambda: _STOP_REQUESTED,
    )
    state = trainer.train(until_unique_tokens=args.until_unique_tokens)
    print(
        "PASS: training stopped " if _STOP_REQUESTED else "PASS: training completed ",
        f"phase={state.phase} steps={state.optimizer_steps} ",
        f"unique_tokens={state.unique_tokens_seen} ",
        f"token_equivalent={state.token_equivalent_compute}",
        sep="",
    )


if __name__ == "__main__":
    main()
