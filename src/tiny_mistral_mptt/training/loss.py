from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def normalize_pass_weights(
    weights: Sequence[float] | None,
    passes: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return right-aligned non-negative pass weights normalized to sum to one.

    A configured vector may be longer than the number of passes sampled for a
    batch. In that case its last ``passes`` entries are used. If it is shorter,
    leading zeroes are inserted. This preserves the older MPTT repository's
    useful convention that later-pass emphasis survives mixed pass-count
    schedules without architecture-specific objective code.
    """
    if passes < 1:
        raise ValueError("passes must be positive")
    if weights is None:
        result = torch.ones(passes, device=device, dtype=dtype)
    else:
        values = [float(value) for value in weights]
        if not values:
            raise ValueError("pass weights must not be empty")
        if len(values) > passes:
            values = values[-passes:]
        elif len(values) < passes:
            values = [0.0] * (passes - len(values)) + values
        result = torch.tensor(values, device=device, dtype=dtype)
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("pass weights must be finite")
    if bool((result < 0).any().item()):
        raise ValueError("pass weights must be non-negative")
    total = result.sum()
    if float(total.detach().cpu()) <= 0:
        raise ValueError("at least one pass weight must be positive")
    return result / total


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("logits [B,T,V] and input_ids [B,T] must align")
    if input_ids.shape[1] < 2:
        raise ValueError("causal LM loss requires at least two tokens")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous().to(logits.device)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
    )
