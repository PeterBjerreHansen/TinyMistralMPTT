from __future__ import annotations

import torch


def mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(spec: str | torch.device = "auto") -> torch.device:
    if isinstance(spec, torch.device):
        device = spec
    elif spec == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif mps_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(spec)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if device.type == "mps" and not mps_available():
        raise RuntimeError(
            "MPS requested but it is unavailable. Use a PyTorch build with MPS support "
            "on a supported Apple Silicon/macOS system."
        )
    return device


def resolve_dtype(spec: str | torch.dtype | None, device: torch.device) -> torch.dtype | None:
    if isinstance(spec, torch.dtype) or spec is None:
        return spec
    if spec == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device.type == "mps":
            # FP16 is the conservative portable default across Apple Silicon
            # machines and PyTorch/macOS versions. Users can request bfloat16.
            return torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[spec]
    except KeyError as exc:
        raise ValueError(f"unknown dtype: {spec}") from exc


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
