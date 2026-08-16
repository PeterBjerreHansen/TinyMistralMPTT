from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .config import MistralConfig, tiny_mistral_248m_config
from .device import resolve_device, resolve_dtype
from .modeling import AttentionBackend, MistralForCausalLM, MistralRotaryEmbedding

MODEL_ID = "M4-ai/TinyMistral-248M-v3"
MODEL_REVISION = "5afbc96ddc964c68282cd970ef49e8d1a5e81c52"
EXPECTED_PARAMETER_COUNT = 248_024_064
EXPECTED_WEIGHTS_SHA256 = "9432ee6e0681473a9ed513e43362d9911832f9a5c7faded76f46ec66c55a9d3b"


def download_snapshot(
    local_dir: str | Path = "checkpoints/TinyMistral-248M-v3",
    *,
    repo_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('install download support with: pip install -e ".[io]"') from exc

    local_dir = Path(local_dir)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )
    return local_dir


def checkpoint_tensor_metadata(weights_path: str | Path) -> dict[str, tuple[tuple[int, ...], str]]:
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_slice(key)
            result[key] = (tuple(tensor.get_shape()), str(tensor.get_dtype()))
    return result


def expected_state_metadata(config: MistralConfig) -> dict[str, tuple[int, ...]]:
    with torch.device("meta"):
        model = MistralForCausalLM(config, attention_backend="reference", initialize=False)
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def verify_checkpoint_structure(model_dir: str | Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    config = MistralConfig.from_json_file(model_dir / "config.json")
    expected = expected_state_metadata(config)
    actual = checkpoint_tensor_metadata(model_dir / "model.safetensors")

    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    shape_mismatches = {
        k: {"expected": expected[k], "actual": actual[k][0]}
        for k in sorted(expected_keys & actual_keys)
        if expected[k] != actual[k][0]
    }
    parameter_count = sum(math_prod(shape) for shape, _dtype in actual.values())
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "parameter_count": parameter_count,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "ok": not missing and not unexpected and not shape_mismatches,
    }


def verify_target_checkpoint(model_dir: str | Path) -> dict[str, Any]:
    """Verify the exact pinned TinyMistral-248M-v3 acceptance target."""
    model_dir = Path(model_dir)
    result = verify_checkpoint_structure(model_dir)
    actual_config = MistralConfig.from_json_file(model_dir / "config.json")
    target_config = tiny_mistral_248m_config()
    config_mismatches = {
        name: {"expected": expected, "actual": actual}
        for name, expected in target_config.to_dict().items()
        if (actual := actual_config.to_dict()[name]) != expected
    }
    parameter_count_matches = result["parameter_count"] == EXPECTED_PARAMETER_COUNT
    weights_sha256 = file_sha256(model_dir / "model.safetensors")
    weights_sha256_matches = weights_sha256 == EXPECTED_WEIGHTS_SHA256
    result.update(
        {
            "config_mismatches": config_mismatches,
            "parameter_count_matches": parameter_count_matches,
            "weights_sha256": weights_sha256,
            "expected_weights_sha256": EXPECTED_WEIGHTS_SHA256,
            "weights_sha256_matches": weights_sha256_matches,
            "ok": (
                result["ok"]
                and not config_mismatches
                and parameter_count_matches
                and weights_sha256_matches
            ),
        }
    )
    return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def math_prod(shape: tuple[int, ...]) -> int:
    out = 1
    for x in shape:
        out *= x
    return out


def load_model(
    model_dir: str | Path,
    *,
    attention_backend: AttentionBackend = "auto",
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype | None = None,
    compile_flex: bool = True,
    flex_block_size: int = 128,
) -> MistralForCausalLM:
    model_dir = Path(model_dir)
    config = MistralConfig.from_json_file(model_dir / "config.json")
    weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    # Construct on meta so strict loading does not transiently allocate a second
    # full randomly initialized copy of the 248M-parameter model.
    with torch.device("meta"):
        model = MistralForCausalLM(
            config,
            attention_backend=attention_backend,
            compile_flex=compile_flex,
            flex_block_size=flex_block_size,
            initialize=False,
        )
    state_dict = load_file(str(weights_path), device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"strict load unexpectedly returned missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state_dict

    # Non-persistent RoPE buffers are intentionally absent from the checkpoint.
    # A model constructed under `torch.device("meta")` therefore still has meta
    # `inv_freq` buffers after assign=True. Materialize those tiny buffers before
    # moving the module tree.
    for module in model.modules():
        if isinstance(module, MistralRotaryEmbedding):
            inv_freq = 1.0 / (
                module.base
                ** (torch.arange(0, module.dim, 2, dtype=torch.int64).float() / module.dim)
            )
            module.inv_freq = inv_freq

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)

    # Cast on CPU before transferring to MPS. Some Apple/PyTorch combinations
    # cannot ingest the checkpoint's native bfloat16 tensors directly, whereas
    # fp16 is broadly supported. This also avoids a transient second full model
    # allocation on the accelerator.
    if resolved_dtype is not None:
        model = model.to(dtype=resolved_dtype)
    model = model.to(device=resolved_device)
    return model
