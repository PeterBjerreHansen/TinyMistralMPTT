from __future__ import annotations

from pathlib import Path

import torch

from tiny_mistral.loading import load_model
from tiny_mistral.modeling import MistralForCausalLM

from .variants import ExperimentalVariant, VanillaVariant


def build_variant(name: str, backbone: MistralForCausalLM) -> ExperimentalVariant:
    if name == "vanilla":
        return VanillaVariant(backbone)
    raise ValueError(
        f"unknown variant {name!r}; this bootstrap intentionally exposes only 'vanilla'"
    )


def load_variant(
    name: str,
    model_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype | None = None,
    attention_backend: str = "auto",
    compile_flex: bool = True,
) -> ExperimentalVariant:
    backbone = load_model(
        model_dir,
        device=device,
        dtype=dtype,
        attention_backend=attention_backend,
        compile_flex=compile_flex,
    )
    return build_variant(name, backbone)
