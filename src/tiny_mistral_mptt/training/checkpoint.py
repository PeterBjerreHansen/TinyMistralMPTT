from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import random
from typing import Any

import torch


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
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler_state,
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
    # These fields identify where/how a resume is invoked, not the training
    # trajectory itself. Every other field must remain identical.
    ignored = {"output_dir", "resume_from"}
    return {key: value for key, value in config.items() if key not in ignored}


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_manifest_sha256: str,
    expected_experiment_config: dict | None = None,
) -> tuple[TrainState, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported experiment checkpoint format")
    if payload["data_manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("data manifest changed across resume")
    if expected_experiment_config is not None:
        recorded = _resume_config_view(payload["experiment_config"])
        requested = _resume_config_view(expected_experiment_config)
        if recorded != requested:
            changed = sorted(
                key for key in set(recorded) | set(requested)
                if recorded.get(key) != requested.get(key)
            )
            raise ValueError(f"experiment config changed across resume: {changed}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_state(payload["rng"])
    return TrainState(**payload["train_state"]), payload["sampler"]
