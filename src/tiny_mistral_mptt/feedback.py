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
    positions: torch.Tensor
    next_sequence_positions: torch.Tensor
    projected_keys: tuple[torch.Tensor, ...] | None = None
    projected_values: tuple[torch.Tensor, ...] | None = None

    def __post_init__(self) -> None:
        if self.memories.ndim != 3:
            raise ValueError("TapeState.memories must be [B,W,D]")
        if self.valid.ndim != 2 or self.valid.shape != self.memories.shape[:2]:
            raise ValueError("TapeState.valid must be bool [B,W]")
        if self.valid.dtype != torch.bool:
            raise ValueError("TapeState.valid must have bool dtype")
        if self.positions.shape != self.valid.shape or self.positions.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("TapeState.positions must be integer [B,W]")
        if self.next_sequence_positions.shape != (self.memories.shape[0],) or (
            self.next_sequence_positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "TapeState.next_sequence_positions must be integer [B]"
            )
        if bool((self.positions[self.valid] < 0).any()):
            raise ValueError("valid TapeState positions must be non-negative")
        if bool((self.next_sequence_positions < 0).any()):
            raise ValueError("next Tape sequence positions must be non-negative")
        if self.memories.shape[1] < 1:
            raise ValueError("TapeState capacity must be positive")
        if (self.projected_keys is None) != (self.projected_values is None):
            raise ValueError("projected_keys and projected_values must be provided together")
        if self.projected_keys is not None:
            assert self.projected_values is not None
            if len(self.projected_keys) != len(self.projected_values):
                raise ValueError("projected K/V tuple lengths differ")
            for key, value in zip(self.projected_keys, self.projected_values, strict=True):
                if key.ndim != 4 or key.shape != value.shape:
                    raise ValueError("projected K/V must have matching [B,Hkv,W,Dh] shapes")
                if key.shape[0] != self.batch_size or key.shape[2] != self.capacity:
                    raise ValueError("projected K/V batch/capacity mismatch")

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
class HybridPassSource:
    """Full-stream sources produced by one tape/recurrent model pass."""

    recurrent_hidden: torch.Tensor
    tape_hidden: torch.Tensor

    def __post_init__(self) -> None:
        if self.recurrent_hidden.ndim != 3 or self.tape_hidden.ndim != 3:
            raise ValueError("hybrid pass sources must be [B,T,D]")
        if self.recurrent_hidden.shape != self.tape_hidden.shape:
            raise ValueError("hybrid recurrent/tape pass sources must have equal shapes")


@dataclass(frozen=True)
class HybridFeedbackState:
    """Immediate recurrent state plus an addressable tape.

    ``fast_hidden`` retains its original name for checkpoint/API compatibility;
    it may contain either a top-layer MemoryAdd state or an internal-layer
    recirculation source.
    """

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

    @property
    def recurrent_hidden(self) -> torch.Tensor:
        return self.fast_hidden


FeedbackMemory = torch.Tensor | TapeState | HybridFeedbackState


def feedback_shape(memory: FeedbackMemory) -> tuple[int, int]:
    """Return ``(batch_size, hidden_size)`` for any feedback-state type."""
    if isinstance(memory, torch.Tensor):
        if memory.ndim != 3:
            raise ValueError("tensor feedback memory must be [B,M,D]")
        return int(memory.shape[0]), int(memory.shape[-1])
    return memory.batch_size, memory.hidden_size
