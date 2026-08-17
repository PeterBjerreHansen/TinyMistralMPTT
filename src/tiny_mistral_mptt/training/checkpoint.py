from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import random
from typing import Any

import torch


_DEFAULT_EXPERIMENT_FIELDS = {
    "prefix_mixin_probability": 0.0,
}

# These one-off FBT calibration controls existed in format-v2 checkpoints but
# never affect a resumed trajectory after model/optimizer state is restored.
# Ignore them so current configs can resume historical checkpoints without
# keeping the retired calibration experiment in the stable config surface.
_LEGACY_NON_TRAJECTORY_FIELDS = {
    "fbt_initialization",
    "fbt_calibration_split",
    "fbt_calibration_block",
    "fbt_gate_logit_std_target",
}


@dataclass
class TrainState:
    optimizer_steps: int = 0
    micro_steps: int = 0
    unique_tokens_seen: int = 0
    token_equivalent_compute: int = 0
    phase: str = "B"


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler_state: dict,
    train_state: TrainState,
    experiment_config: dict,
    data_manifest_sha256: str,
    pass_scheduler_state: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler_state,
        "pass_scheduler": pass_scheduler_state,
        "train_state": asdict(train_state),
        "rng": capture_rng_state(),
        "experiment_config": experiment_config,
        "data_manifest_sha256": data_manifest_sha256,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def _resume_config_view(config: dict) -> dict:
    # These fields identify where/how a resume is invoked, not the trajectory.
    # ``max_unique_tokens`` is a stopping budget, so allowing it to increase is
    # what makes an exact frozen-phase continuation possible.
    ignored = {
        "output_dir",
        "resume_from",
        "max_unique_tokens",
        "init_from",
        "eval_every_tokens",
        "eval_batches",
        "eval_passes",
        "checkpoint_every_tokens",
        *_LEGACY_NON_TRAJECTORY_FIELDS,
    }
    result = {
        key: value for key, value in config.items() if key not in ignored
    }
    for key, default in _DEFAULT_EXPERIMENT_FIELDS.items():
        result.setdefault(key, default)
    return result


def load_model_weights(path: str | Path, *, model: torch.nn.Module) -> dict[str, Any]:
    """Load model parameters only from an experiment checkpoint for ``init_from``."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") not in {1, 2}:
        raise ValueError("unsupported experiment checkpoint format")
    model.load_state_dict(payload["model"], strict=True)
    return {
        "source_path": str(path),
        "source_train_state": payload.get("train_state"),
        "source_experiment_config": payload.get("experiment_config"),
    }


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_manifest_sha256: str,
    expected_experiment_config: dict | None = None,
    pass_scheduler=None,
) -> tuple[TrainState, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = payload.get("format_version")
    if version not in {1, 2}:
        raise ValueError("unsupported experiment checkpoint format")
    if payload["data_manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("data manifest changed across resume")
    if expected_experiment_config is not None:
        recorded = _resume_config_view(payload["experiment_config"])
        requested = _resume_config_view(expected_experiment_config)
        if version == 1:
            # Bootstrap checkpoints predate the newly added multipass fields.
            # Compare every trajectory field they actually recorded while
            # allowing new default-only fields to be absent. Pass-scheduler
            # compatibility is checked separately below.
            changed = sorted(
                key for key, value in recorded.items()
                if requested.get(key) != value
            )
        else:
            changed = sorted(
                key for key in set(recorded) | set(requested)
                if recorded.get(key) != requested.get(key)
            )
        if changed:
            raise ValueError(f"experiment config changed across resume: {changed}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if pass_scheduler is not None:
        scheduler_state = payload.get("pass_scheduler")
        if scheduler_state is None:
            # Version-1 vanilla checkpoints predate pass scheduling. They are
            # compatible only with the implicit fixed one-pass schedule.
            if pass_scheduler.stages != [{"until_tokens": None, "probabilities": {1: 1.0}}]:
                raise ValueError("checkpoint predates pass scheduler and is not compatible with this schedule")
        else:
            pass_scheduler.load_state_dict(scheduler_state)
    restore_rng_state(payload["rng"])
    return TrainState(**payload["train_state"]), payload["sampler"]
