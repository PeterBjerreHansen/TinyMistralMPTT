from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TapeState:
    """Fixed-capacity chronological memory bank for cached inference.

    ``memories`` has shape ``[B,W,D]`` and ``valid`` has shape ``[B,W]``.
    Valid entries are kept left-aligned in chronological order.  The fixed
    capacity makes per-example token-triggered writes batchable even when
    examples have different write counts.
    """

    memories: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.memories.ndim != 3:
            raise ValueError("TapeState.memories must be [B,W,D]")
        if self.valid.ndim != 2 or self.valid.shape != self.memories.shape[:2]:
            raise ValueError("TapeState.valid must be bool [B,W]")
        if self.valid.dtype != torch.bool:
            raise ValueError("TapeState.valid must have bool dtype")
        if self.memories.shape[1] < 1:
            raise ValueError("TapeState capacity must be positive")

    @property
    def batch_size(self) -> int:
        return int(self.memories.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.memories.shape[-1])

    @property
    def capacity(self) -> int:
        return int(self.memories.shape[1])


@dataclass(frozen=True)
class HybridFeedbackState:
    """Immediate MemoryAdd state plus a addressable tape."""

    fast_hidden: torch.Tensor
    tape: TapeState

    def __post_init__(self) -> None:
        if self.fast_hidden.ndim != 3 or self.fast_hidden.shape[1] != 1:
            raise ValueError("HybridFeedbackState.fast_hidden must be [B,1,D]")
        if self.fast_hidden.shape[0] != self.tape.batch_size:
            raise ValueError("hybrid fast/tape batch sizes differ")
        if self.fast_hidden.shape[-1] != self.tape.hidden_size:
            raise ValueError("hybrid fast/tape hidden dimensions differ")

    @property
    def batch_size(self) -> int:
        return int(self.fast_hidden.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.fast_hidden.shape[-1])


FeedbackMemory = torch.Tensor | TapeState | HybridFeedbackState


def feedback_shape(memory: FeedbackMemory) -> tuple[int, int]:
    """Return ``(batch_size, hidden_size)`` for any feedback-state type."""
    if isinstance(memory, torch.Tensor):
        if memory.ndim != 3:
            raise ValueError("tensor feedback memory must be [B,M,D]")
        return int(memory.shape[0]), int(memory.shape[-1])
    return memory.batch_size, memory.hidden_size
