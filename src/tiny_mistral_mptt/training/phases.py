from __future__ import annotations

from dataclasses import dataclass

from ..variants.base import ExperimentalVariant


@dataclass(frozen=True)
class PhasePlan:
    name: str
    token_budget: int

    def validate(self) -> None:
        if self.name not in {"A", "B"}:
            raise ValueError("phase name must be A or B")
        if self.token_budget < 0:
            raise ValueError("phase token budget must be non-negative")


def configure_phase(model: ExperimentalVariant, phase: str) -> int:
    """Apply phase trainability and return the trainable parameter count."""
    model.set_phase(phase)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def vanilla_phase_a_is_noop(model: ExperimentalVariant) -> bool:
    return not any(True for _ in model.added_parameters())
