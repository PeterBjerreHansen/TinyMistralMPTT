from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import re
from typing import Any

import torch


_DEFAULT_EXPERIMENT_FIELDS = {
    "prefix_mixin_probability": 0.0,
    "pass_loss_weights_by_k": None,
    "autocast_dtype": None,
    "checkpoint_every_seconds": 0.0,
    "checkpoint_keep_last": 2,
    "snapshot_at_tokens": None,
}

_LEGACY_NON_TRAJECTORY_FIELDS = {
    "fbt_initialization",
    "fbt_calibration_split",
    "fbt_calibration_block",
    "fbt_gate_logit_std_target",
}

_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d{12})\.pt$")


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


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_replace_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _durable_replace_symlink(path: Path, target: str) -> None:
    """Atomically replace a small compatibility link after its target is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    os.symlink(target, temporary)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler_state: dict,
    train_state: TrainState,
    experiment_config: dict,
    data_manifest_sha256: str,
    pass_scheduler_state: dict[str, Any] | None,
    source_provenance: dict[str, Any] | None,
    checkpoint_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format_version": 3,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler_state,
        "pass_scheduler": pass_scheduler_state,
        "train_state": asdict(train_state),
        "rng": capture_rng_state(),
        "experiment_config": experiment_config,
        "data_manifest_sha256": data_manifest_sha256,
        "source_provenance": source_provenance,
        "checkpoint_metadata": checkpoint_metadata or {},
    }


def _save_payload_durable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


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
    source_provenance: dict[str, Any] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one durable checkpoint.

    This compatibility entry point remains useful for tests and historical
    callers. Serious runs should use ``save_checkpoint_generation`` so a
    previous known-good generation remains available during replacement.
    """
    path = Path(path)
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        sampler_state=sampler_state,
        train_state=train_state,
        experiment_config=experiment_config,
        data_manifest_sha256=data_manifest_sha256,
        pass_scheduler_state=pass_scheduler_state,
        source_provenance=source_provenance,
        checkpoint_metadata=checkpoint_metadata,
    )
    _save_payload_durable(path, payload)
    return path


def checkpoint_filename(unique_tokens_seen: int) -> str:
    tokens = int(unique_tokens_seen)
    if tokens < 0:
        raise ValueError("unique_tokens_seen must be non-negative")
    return f"checkpoint_{tokens:012d}.pt"


def checkpoint_directory(run_dir: str | Path) -> Path:
    return Path(run_dir) / "checkpoints"


def discover_checkpoint_generations(run_dir: str | Path) -> list[Path]:
    directory = checkpoint_directory(run_dir)
    if not directory.exists():
        return []
    candidates: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = _CHECKPOINT_RE.match(path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in candidates]


def _read_latest_pointer(run_dir: str | Path) -> dict[str, Any] | None:
    path = checkpoint_directory(run_dir) / "latest.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def candidate_checkpoint_paths(run_dir: str | Path) -> list[Path]:
    directory = checkpoint_directory(run_dir)
    pointer = _read_latest_pointer(run_dir)
    ordered: list[Path] = []
    if pointer:
        for key in ("current", "previous"):
            name = pointer.get(key)
            if isinstance(name, str):
                path = directory / name
                if path not in ordered:
                    ordered.append(path)
    for path in discover_checkpoint_generations(run_dir):
        if path not in ordered:
            ordered.append(path)
    return ordered


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = payload.get("format_version")
    if version not in {1, 2, 3}:
        raise ValueError("unsupported experiment checkpoint format")
    required = {"model", "optimizer", "sampler", "train_state", "rng", "data_manifest_sha256"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint missing required fields: {missing}")
    state = TrainState(**payload["train_state"])
    return {
        "format_version": int(version),
        "train_state": asdict(state),
        "experiment_config": payload.get("experiment_config"),
        "data_manifest_sha256": payload["data_manifest_sha256"],
        "source_provenance": payload.get("source_provenance"),
        "checkpoint_metadata": payload.get("checkpoint_metadata") or {},
    }


def save_checkpoint_generation(
    run_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler_state: dict,
    train_state: TrainState,
    experiment_config: dict,
    data_manifest_sha256: str,
    pass_scheduler_state: dict[str, Any] | None = None,
    source_provenance: dict[str, Any] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
    keep_last: int = 2,
) -> Path:
    if int(keep_last) < 2:
        raise ValueError("generation checkpointing requires keep_last>=2")
    directory = checkpoint_directory(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / checkpoint_filename(train_state.unique_tokens_seen)
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        sampler_state=sampler_state,
        train_state=train_state,
        experiment_config=experiment_config,
        data_manifest_sha256=data_manifest_sha256,
        pass_scheduler_state=pass_scheduler_state,
        source_provenance=source_provenance,
        checkpoint_metadata=checkpoint_metadata,
    )
    _save_payload_durable(path, payload)

    # Re-open before advertising the new generation as current. This catches
    # serialization/truncation failures while the previous pointer is intact.
    metadata = inspect_checkpoint(path)
    if int(metadata["train_state"]["unique_tokens_seen"]) != train_state.unique_tokens_seen:
        raise RuntimeError("checkpoint verification returned the wrong token count")

    generations = discover_checkpoint_generations(run_dir)
    current = path.name
    previous = next((candidate.name for candidate in generations if candidate != path), None)
    pointer = {
        "format_version": 1,
        "current": current,
        "previous": previous,
        "unique_tokens_seen": int(train_state.unique_tokens_seen),
        "optimizer_steps": int(train_state.optimizer_steps),
    }
    _durable_replace_bytes(
        directory / "latest.json",
        (json.dumps(pointer, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    # Keep the historical run-dir path usable for explicit ``resume_from`` and
    # ``init_from`` callers. Auto-resume uses latest.json as its source of
    # truth, while this small link avoids duplicating the checkpoint payload.
    _durable_replace_symlink(
        Path(run_dir) / "latest.pt",
        os.path.relpath(path, start=Path(run_dir)),
    )

    # Only prune after both the checkpoint and pointer are durable.
    for candidate in generations[int(keep_last) :]:
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
    _fsync_directory(directory)
    return path


def _resume_config_view(config: dict) -> dict:
    ignored = {
        "model_dir",
        "data_dir",
        "output_dir",
        "resume_from",
        "init_from",
        "eval_every_tokens",
        "eval_batches",
        "eval_passes",
        "checkpoint_every_tokens",
        "checkpoint_every_seconds",
        "checkpoint_keep_last",
        "snapshot_at_tokens",
        *_LEGACY_NON_TRAJECTORY_FIELDS,
    }
    schedule = config.get("lr_schedule")
    schedule_type = "cosine" if schedule is None else str(schedule.get("type", "cosine"))
    if schedule_type == "constant":
        ignored.add("max_unique_tokens")
    result = {key: value for key, value in config.items() if key not in ignored}
    for key, default in _DEFAULT_EXPERIMENT_FIELDS.items():
        result.setdefault(key, default)
    for key in ("checkpoint_every_seconds", "checkpoint_keep_last", "snapshot_at_tokens"):
        result.pop(key, None)
    return result


def _source_identity(source: dict[str, Any] | None) -> tuple[Any, Any, Any]:
    if not source:
        return (None, None, None)
    code_hash = source.get("source_code_sha256")
    uv_hash = source.get("uv_lock_sha256")
    if code_hash is not None:
        return ("code", code_hash, uv_hash)
    return ("git", source.get("git_commit"), uv_hash)

def _validate_payload(
    payload: dict[str, Any],
    *,
    expected_manifest_sha256: str,
    expected_experiment_config: dict | None,
    expected_source_provenance: dict[str, Any] | None,
    allow_source_mismatch: bool,
    pass_scheduler=None,
) -> None:
    version = payload.get("format_version")
    if version not in {1, 2, 3}:
        raise ValueError("unsupported experiment checkpoint format")
    if payload["data_manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("data manifest changed across resume")
    if expected_experiment_config is not None:
        recorded = _resume_config_view(payload["experiment_config"])
        requested = _resume_config_view(expected_experiment_config)
        if version == 1:
            changed = sorted(
                key for key, value in recorded.items() if requested.get(key) != value
            )
        else:
            changed = sorted(
                key for key in set(recorded) | set(requested)
                if recorded.get(key) != requested.get(key)
            )
        if changed:
            raise ValueError(f"experiment config changed across resume: {changed}")
    if version >= 3 and expected_source_provenance is not None and not allow_source_mismatch:
        recorded_source = payload.get("source_provenance")
        if _source_identity(recorded_source) != _source_identity(expected_source_provenance):
            raise ValueError("source commit, uv.lock, or execution-code hash changed across resume")
    if pass_scheduler is not None:
        scheduler_state = payload.get("pass_scheduler")
        if scheduler_state is None:
            if pass_scheduler.stages != [{"until_tokens": None, "probabilities": {1: 1.0}}]:
                raise ValueError("checkpoint predates pass scheduler and is not compatible with this schedule")
        else:
            # Validate on a throwaway logical state is not available here; the
            # actual scheduler load happens only after all other checks pass.
            recorded_stages = scheduler_state.get("stages")
            if recorded_stages is not None and recorded_stages != pass_scheduler.stages:
                raise ValueError("pass schedule changed across resume")


def load_model_weights(path: str | Path, *, model: torch.nn.Module) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") not in {1, 2, 3}:
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
    expected_source_provenance: dict[str, Any] | None = None,
    allow_source_mismatch: bool = False,
    pass_scheduler=None,
) -> tuple[TrainState, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    _validate_payload(
        payload,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_experiment_config=expected_experiment_config,
        expected_source_provenance=expected_source_provenance,
        allow_source_mismatch=allow_source_mismatch,
        pass_scheduler=pass_scheduler,
    )
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if pass_scheduler is not None:
        scheduler_state = payload.get("pass_scheduler")
        if scheduler_state is not None:
            pass_scheduler.load_state_dict(scheduler_state)
    restore_rng_state(payload["rng"])
    return TrainState(**payload["train_state"]), payload["sampler"]


def load_latest_valid_checkpoint(
    run_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_manifest_sha256: str,
    expected_experiment_config: dict | None = None,
    expected_source_provenance: dict[str, Any] | None = None,
    allow_source_mismatch: bool = False,
    pass_scheduler=None,
) -> tuple[Path, TrainState, dict, bool]:
    candidates = candidate_checkpoint_paths(run_dir)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint generations found under {checkpoint_directory(run_dir)}")
    errors: list[str] = []
    for index, path in enumerate(candidates):
        if not path.exists():
            errors.append(f"{path.name}: missing")
            continue
        try:
            # Inspect before mutating live state. Corrupt/truncated files are
            # rejected here and therefore cannot partially modify the model.
            inspect_checkpoint(path)
            state, sampler = load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                expected_manifest_sha256=expected_manifest_sha256,
                expected_experiment_config=expected_experiment_config,
                expected_source_provenance=expected_source_provenance,
                allow_source_mismatch=allow_source_mismatch,
                pass_scheduler=pass_scheduler,
            )
            return path, state, sampler, index > 0
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    detail = "; ".join(errors)
    raise RuntimeError(f"no valid compatible checkpoint generation found: {detail}")
