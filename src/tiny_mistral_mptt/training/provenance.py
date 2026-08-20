from __future__ import annotations

import hashlib
import os
from pathlib import Path
import socket
import subprocess
from typing import Any

import torch


def file_sha256_optional(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_code_sha256(repository: str | Path) -> str:
    """Hash executable Python plus the locked Python environment."""
    repository = Path(repository)
    digest = hashlib.sha256()
    candidates: list[Path] = []
    for root_name in ("src", "scripts"):
        root = repository / root_name
        if root.exists():
            candidates.extend(path for path in root.rglob("*.py") if path.is_file())
    for name in ("pyproject.toml", "uv.lock"):
        path = repository / name
        if path.exists() and path.is_file():
            candidates.append(path)
    for path in sorted(candidates, key=lambda item: item.relative_to(repository).as_posix()):
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def source_provenance(repository: str | Path) -> dict[str, Any]:
    repository = Path(repository)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "uv_lock_sha256": file_sha256_optional(repository / "uv.lock"),
        "source_code_sha256": source_code_sha256(repository),
    }


def hardware_provenance(device: torch.device | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "provider_instance_id": os.environ.get("VERDA_INSTANCE_ID")
        or os.environ.get("RUNPOD_POD_ID")
        or os.environ.get("INSTANCE_ID"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device) if device is not None else None,
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        result.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_bytes": int(props.total_memory),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return result
