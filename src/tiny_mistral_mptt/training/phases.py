from __future__ import annotations

from ..variants.base import ExperimentalVariant


def configure_phase(model: ExperimentalVariant, phase: str) -> int:
    """Apply phase trainability and return the trainable parameter count."""
    model.set_phase(phase)
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
