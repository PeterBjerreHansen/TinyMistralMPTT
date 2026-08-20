from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from tiny_mistral.loading import load_model
from tiny_mistral.modeling import MistralForCausalLM

from .variants import (
    ExperimentalVariant,
    FBTVariant,
    MemoryAddVariant,
    TapeAddHybridVariant,
    TapeVariant,
    VanillaVariant,
)

if TYPE_CHECKING:
    from .config import ExperimentConfig


def build_variant(
    name: str,
    backbone: MistralForCausalLM,
    *,
    architecture_seed: int = 4242,
    memory_window: int = 32,
    memory_write_mode: str | None = None,
    memory_write_stride: int | None = None,
    memory_token_visibility: str | None = None,
    prefix_mixin_probability: float = 0.0,
) -> ExperimentalVariant:
    if name == "vanilla":
        variant: ExperimentalVariant = VanillaVariant(backbone)
    elif name == "fbt":
        variant = FBTVariant(
            backbone,
            initialization_seed=architecture_seed,
            prefix_mixin_probability=prefix_mixin_probability,
        )
    elif name == "memory_add":
        variant = MemoryAddVariant(backbone)
    elif name in {"tape", "tape_add_hybrid"}:
        if memory_write_mode not in {"dense", "periodic", "memory_token"}:
            raise ValueError("tape variants require memory_write_mode: dense|periodic|memory_token")
        if memory_write_mode == "dense":
            if memory_write_stride is not None:
                raise ValueError("dense tape must not set memory_write_stride")
            if memory_token_visibility is not None:
                raise ValueError("dense tape must not set memory_token_visibility")
            stride = 1
            visibility = "visible"
        elif memory_write_mode == "periodic":
            if memory_write_stride is None or int(memory_write_stride) <= 0:
                raise ValueError("periodic tape requires positive memory_write_stride")
            if memory_token_visibility is not None:
                raise ValueError("memory_token_visibility applies only to memory_token mode")
            stride = int(memory_write_stride)
            visibility = "visible"
        else:
            if memory_write_stride is None or int(memory_write_stride) <= 0:
                raise ValueError("memory_token tape requires positive memory_write_stride")
            if memory_token_visibility not in {"visible", "write_only"}:
                raise ValueError("memory_token tape requires memory_token_visibility: visible|write_only")
            stride = int(memory_write_stride)
            visibility = str(memory_token_visibility)
        kwargs = dict(
            memory_window=memory_window,
            memory_write_mode=memory_write_mode,
            memory_write_stride=stride,
            memory_token_visibility=visibility,
            initialization_seed=architecture_seed,
        )
        variant = TapeVariant(backbone, **kwargs) if name == "tape" else TapeAddHybridVariant(backbone, **kwargs)
    else:
        raise ValueError(f"unknown variant {name!r}")

    reference_parameter = next(backbone.parameters())
    variant.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
    return variant


def load_variant(
    name: str,
    model_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype | None = None,
    attention_backend: str = "auto",
    compile_flex: bool = True,
    architecture_seed: int = 4242,
    memory_window: int = 32,
    memory_write_mode: str | None = None,
    memory_write_stride: int | None = None,
    memory_token_visibility: str | None = None,
    prefix_mixin_probability: float = 0.0,
) -> ExperimentalVariant:
    backbone = load_model(
        model_dir,
        device=device,
        dtype=dtype,
        attention_backend=attention_backend,
        compile_flex=compile_flex,
    )
    return build_variant(
        name,
        backbone,
        architecture_seed=architecture_seed,
        memory_window=memory_window,
        memory_write_mode=memory_write_mode,
        memory_write_stride=memory_write_stride,
        memory_token_visibility=memory_token_visibility,
        prefix_mixin_probability=prefix_mixin_probability,
    )


def load_variant_from_config(
    cfg: "ExperimentConfig",
    *,
    device: str | torch.device | None = None,
) -> ExperimentalVariant:
    return load_variant(
        cfg.variant,
        cfg.model_dir,
        device=cfg.device if device is None else device,
        dtype=cfg.dtype,
        attention_backend=cfg.attention_backend,
        architecture_seed=cfg.architecture_seed,
        memory_window=cfg.memory_window,
        memory_write_mode=cfg.memory_write_mode,
        memory_write_stride=cfg.memory_write_stride,
        memory_token_visibility=cfg.memory_token_visibility,
        prefix_mixin_probability=cfg.prefix_mixin_probability,
    )
