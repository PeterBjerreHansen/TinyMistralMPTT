#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.manifest import file_sha256, verify_artifact
from tiny_mistral.loading import verify_target_checkpoint


def _git_state(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Provider-agnostic preflight for a CUDA training run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = _repo_path(root, args.config)
    cfg = load_experiment_config(config_path)
    failures: list[str] = []

    try:
        device = resolve_device(cfg.device)
    except RuntimeError as exc:
        # Keep preflight diagnostic: a local Mac should still receive the
        # checkpoint, data, and provenance checks instead of a traceback.
        device = None
        failures.append(str(exc))

    commit, dirty = _git_state(root)
    if commit is None:
        failures.append("could not resolve git commit")
    if dirty and not args.allow_dirty:
        failures.append("git worktree is dirty")
    if device is None or device.type != "cuda":
        failures.append(f"cloud preflight requires CUDA; resolved {device}")
    if (
        device is not None
        and device.type == "cuda"
        and cfg.autocast_dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        failures.append("config requests BF16 autocast but GPU reports no BF16 support")

    model_dir = _repo_path(root, cfg.model_dir)
    model_verification = None
    if not model_dir.exists():
        failures.append(f"model_dir does not exist: {model_dir}")
    else:
        try:
            model_verification = verify_target_checkpoint(model_dir)
            if not model_verification["ok"]:
                failures.append("model checkpoint does not match the pinned TinyMistral target")
        except Exception as exc:  # preflight should aggregate failures
            failures.append(f"model checkpoint verification failed: {exc}")

    data_dir = _repo_path(root, cfg.data_dir)
    manifest_sha256 = None
    if not data_dir.exists():
        failures.append(f"data_dir does not exist: {data_dir}")
    else:
        try:
            verify_artifact(data_dir)
            manifest_sha256 = file_sha256(data_dir / "manifest.json")
        except Exception as exc:  # preflight should aggregate failures
            failures.append(f"data artifact verification failed: {exc}")

    if cfg.init_from and not _repo_path(root, cfg.init_from).exists():
        failures.append(f"init_from checkpoint does not exist: {cfg.init_from}")
    if cfg.resume_from and not _repo_path(root, cfg.resume_from).exists():
        failures.append(f"resume_from checkpoint does not exist: {cfg.resume_from}")

    output_dir = _repo_path(root, cfg.output_dir)
    if output_dir.exists() and not args.allow_existing_output:
        occupied = any((output_dir / name).exists() for name in ("run.json", "metrics.jsonl", "latest.pt"))
        if occupied:
            failures.append(f"output_dir already contains run artifacts: {output_dir}")

    hardware = {
        "device": str(device) if device is not None else f"unresolved ({cfg.device})",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device is not None and device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        hardware.update(
            {
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )

    report = {
        "status": "pass" if not failures else "fail",
        "config": str(config_path),
        "source": {"git_commit": commit, "git_dirty": dirty},
        "precision": {"parameter_dtype": cfg.dtype, "autocast_dtype": cfg.autocast_dtype},
        "model_verification": model_verification,
        "hardware": hardware,
        "data_manifest_sha256": manifest_sha256,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
