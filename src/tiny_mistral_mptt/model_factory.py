from __future__ import annotations

from pathlib import Path

import torch

from tiny_mistral.loading import load_model
from tiny_mistral.modeling import MistralForCausalLM

from .variants import ExperimentalVariant, FBTVariant, MemoryTape32Variant, VanillaVariant


def build_variant(
    name: str,
    backbone: MistralForCausalLM,
    *,
    architecture_seed: int = 4242,
    memory_window: int = 32,
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
    elif name == "memory_tape32":
        variant = MemoryTape32Variant(
            backbone,
            memory_window=memory_window,
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
        prefix_mixin_probability=prefix_mixin_probability,
    )
