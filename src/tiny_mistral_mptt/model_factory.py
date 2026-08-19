from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from tiny_mistral.loading import load_model
from tiny_mistral.modeling import MistralForCausalLM

from .variants import (
    ExperimentalVariant,
    FBTVariant,
    MemoryAddSparseTapeVariant,
    MemoryAddVariant,
    MemoryTape32Variant,
    SparseMemoryTapeVariant,
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
    memory_write_mode: str = "periodic",
    memory_write_stride: int = 8,
    memory_token_id: int | None = None,
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
    elif name == "memory_tape32":
        variant = MemoryTape32Variant(
            backbone,
            memory_window=memory_window,
            initialization_seed=architecture_seed,
        )
    elif name == "sparse_memory_tape":
        variant = SparseMemoryTapeVariant(
            backbone,
            memory_window=memory_window,
            memory_write_mode=memory_write_mode,
            memory_write_stride=memory_write_stride,
            memory_token_id=memory_token_id,
            initialization_seed=architecture_seed,
        )
    elif name == "memory_add_sparse_tape":
        variant = MemoryAddSparseTapeVariant(
            backbone,
            memory_window=memory_window,
            memory_write_mode=memory_write_mode,
            memory_write_stride=memory_write_stride,
            memory_token_id=memory_token_id,
            initialization_seed=architecture_seed,
        )
    else:
        raise ValueError(f"unknown variant {name!r}")

    # The checkpoint loader has already placed/cast the backbone. Newly created
    # research modules are ordinary CPU FP32 modules, so align them once here.
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
    memory_write_mode: str = "periodic",
    memory_write_stride: int = 8,
    memory_token_id: int | None = None,
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
        memory_token_id=memory_token_id,
        prefix_mixin_probability=prefix_mixin_probability,
    )


def load_variant_from_config(
    cfg: "ExperimentConfig",
    *,
    device: str | torch.device | None = None,
) -> ExperimentalVariant:
    """Construct a variant with every architecture knob carried by a config."""
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
        memory_token_id=cfg.memory_token_id,
        prefix_mixin_probability=cfg.prefix_mixin_probability,
    )
