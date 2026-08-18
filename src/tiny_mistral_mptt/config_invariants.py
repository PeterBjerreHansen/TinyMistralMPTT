"""Execution settings that must match across comparable experiment arms."""

from __future__ import annotations

from typing import Any

from .config import ExperimentConfig


# These fields describe the execution environment and optimization trajectory.
# Experiment axes such as pass_schedule, pass-loss weights, and eval_passes are
# intentionally checked separately by the protocol-specific gate.
EXECUTION_INVARIANT_FIELDS = (
    "variant",
    "phase",
    "model_dir",
    "data_dir",
    "device",
    "dtype",
    "autocast_dtype",
    "attention_backend",
    "seed",
    "architecture_seed",
    "batch_size",
    "grad_accum_steps",
    "max_unique_tokens",
    "learning_rate",
    "pretrained_learning_rate",
    "added_learning_rate",
    "min_lr_ratio",
    "warmup_tokens",
    "lr_schedule",
    "weight_decay",
    "grad_clip",
    "eval_every_tokens",
    "eval_batches",
    "checkpoint_every_tokens",
    "memory_window",
    "prefix_mixin_probability",
)


def execution_invariant_view(config: ExperimentConfig) -> dict[str, Any]:
    """Return normalized execution settings suitable for equality checks."""
    return {field: getattr(config, field) for field in EXECUTION_INVARIANT_FIELDS}


def differing_invariant_fields(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    """Return invariant names whose values differ between two config views."""
    return sorted(
        field
        for field in set(reference) | set(candidate)
        if reference.get(field) != candidate.get(field)
    )
