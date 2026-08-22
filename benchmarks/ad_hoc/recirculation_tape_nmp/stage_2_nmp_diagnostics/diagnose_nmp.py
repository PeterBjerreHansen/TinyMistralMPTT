#!/usr/bin/env python
"""Measure NTP and NMP scales at a checkpoint without updating the model."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import statistics

import torch

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.training.checkpoint import load_model_weights


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("could not locate repository root from diagnostic script")


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _repo_root() / candidate


def _mean(records: list[dict[str, float]], key: str) -> float | None:
    values = [float(record[key]) for record in records if key in record]
    return statistics.fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure checkpoint NTP/NMP losses without optimizer updates."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="optional model-only checkpoint override, useful before the 10M parent exists",
    )
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.batches <= 0 or args.start_index < 0 or args.passes < 1:
        raise SystemExit("batches and passes must be positive; start-index must be non-negative")

    config_path = _resolve_repo_path(args.config)
    cfg = load_experiment_config(config_path)
    checkpoint = _resolve_repo_path(args.checkpoint or cfg.init_from or "")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"diagnostic checkpoint does not exist: {checkpoint}")
    cfg.init_from = str(checkpoint)
    cfg.resume_from = None
    cfg.validate()
    if args.passes not in {
        passes
        for stage in cfg.normalized_pass_schedule()
        for passes in stage["probabilities"]
    }:
        raise ValueError(f"passes={args.passes} is not represented in the config pass schedule")

    device = resolve_device(cfg.device)
    model = load_variant_from_config(cfg, device=device)
    provenance = load_model_weights(checkpoint, model=model)
    model.eval()
    data = load_packed_dataset_for_experiment(
        cfg.data_dir,
        "train",
        memory_write_mode=cfg.memory_write_mode,
        memory_write_stride=cfg.memory_write_stride,
    )
    if args.start_index >= len(data):
        raise ValueError(f"start-index {args.start_index} is outside {len(data)} training blocks")

    records: list[dict[str, float]] = []
    with torch.inference_mode():
        for offset in range(args.batches):
            index = args.start_index + offset
            if index >= len(data):
                break
            ids = data.batch([index], device=device)
            output = model.compute_loss(
                ids,
                phase=cfg.phase,
                passes=args.passes,
                loss_weights=cfg.ntp_loss_weights_for_passes(args.passes),
                recurrent_nmp_loss_weights=cfg.recurrent_nmp_loss_weights_for_passes(
                    args.passes
                ),
                tape_nmp_loss_weights=cfg.tape_nmp_loss_weights_for_passes(
                    args.passes
                ),
                nmp_weight_scale=1.0,
            )
            records.append(dict(output.metrics))
    if not records:
        raise RuntimeError("diagnostic selected no batches")

    report: dict[str, object] = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_train_state": provenance["source_train_state"],
        "device": str(device),
        "passes": args.passes,
        "batches": len(records),
        "metrics": {},
    }
    metrics: dict[str, float] = {}
    keys = sorted({key for record in records for key in record})
    for key in keys:
        value = _mean(records, key)
        if value is not None and math.isfinite(value):
            metrics[key] = value
    ntp = metrics.get("ntp_loss")
    recurrent = metrics.get("recurrent_nmp_loss")
    tape = metrics.get("tape_nmp_loss")
    if ntp is not None and ntp > 0:
        if recurrent is not None:
            metrics["recurrent_nmp_to_ntp_ratio"] = recurrent / ntp
        if tape is not None:
            metrics["tape_nmp_to_ntp_ratio"] = tape / ntp
        weighted = sum(
            metrics.get(key, 0.0)
            for key in ("recurrent_nmp_weighted_loss", "tape_nmp_weighted_loss")
        )
        metrics["weighted_nmp_to_ntp_ratio"] = weighted / ntp
    report["metrics"] = metrics

    output_path = (
        _resolve_repo_path(args.output)
        if args.output is not None
        else _resolve_repo_path(cfg.output_dir) / f"diagnostic_k{args.passes}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
