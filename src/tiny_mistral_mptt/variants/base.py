from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn


@dataclass
class TrainOutput:
    loss: torch.Tensor
    pass_losses: tuple[torch.Tensor, ...]
    effective_passes: int
    metrics: dict[str, float] = field(default_factory=dict)


class ExperimentalVariant(nn.Module):
    """Small trainer-facing contract shared by vanilla and multipass variants."""

    variant_name: str

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        *,
        phase: str = "B",
        passes: int = 1,
        loss_weights: Sequence[float] | None = None,
    ) -> TrainOutput:
        raise NotImplementedError

    def added_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters absent from the validated vanilla backbone."""
        return ()

    def pretrained_parameters(self) -> Iterable[nn.Parameter]:
        added_ids = {id(parameter) for parameter in self.added_parameters()}
        return (parameter for parameter in self.parameters() if id(parameter) not in added_ids)

    def set_phase(self, phase: str) -> None:
        if phase not in {"A", "B"}:
            raise ValueError("phase must be 'A' or 'B'")
        if phase == "A":
            added_ids = {id(parameter) for parameter in self.added_parameters()}
            for parameter in self.parameters():
                parameter.requires_grad_(id(parameter) in added_ids)
        else:
            for parameter in self.parameters():
                parameter.requires_grad_(True)
