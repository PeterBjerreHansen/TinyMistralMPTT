from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class TrainOutput:
    loss: torch.Tensor
    pass_losses: tuple[torch.Tensor, ...]
    effective_passes: int
    metrics: dict[str, float] = field(default_factory=dict)


class ExperimentalVariant(nn.Module):
    """Small trainer-facing contract; intentionally not an MPTT framework yet."""

    variant_name: str

    def compute_loss(self, input_ids: torch.Tensor, *, phase: str = "B", passes: int = 1) -> TrainOutput:
        raise NotImplementedError

    def added_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters absent from the validated vanilla backbone."""
        return ()

    def set_phase(self, phase: str) -> None:
        if phase not in {"A", "B"}:
            raise ValueError("phase must be 'A' or 'B'")
        if phase == "A":
            added_ids = {id(p) for p in self.added_parameters()}
            for parameter in self.parameters():
                parameter.requires_grad_(id(parameter) in added_ids)
        else:
            for parameter in self.parameters():
                parameter.requires_grad_(True)
