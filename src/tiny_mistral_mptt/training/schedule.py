from __future__ import annotations

import math
from itertools import pairwise
from typing import Any


def cosine_lr_multiplier(
    tokens_seen: int,
    *,
    total_tokens: int,
    warmup_tokens: int,
    min_lr_ratio: float,
) -> float:
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if warmup_tokens < 0:
        raise ValueError("warmup_tokens must be non-negative")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if warmup_tokens and tokens_seen < warmup_tokens:
        return max(tokens_seen, 1) / warmup_tokens
    if total_tokens <= warmup_tokens:
        return 1.0
    progress = (tokens_seen - warmup_tokens) / (total_tokens - warmup_tokens)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def piecewise_linear_multiplier(
    tokens_seen: int, points: list[list[float] | tuple[float, float]]
) -> float:
    if tokens_seen < 0:
        raise ValueError("tokens_seen must be non-negative")
    parsed = [(int(point[0]), float(point[1])) for point in points]
    if tokens_seen <= parsed[0][0]:
        return parsed[0][1]
    for (left_t, left_y), (right_t, right_y) in pairwise(parsed):
        if tokens_seen <= right_t:
            if right_t == left_t:
                return right_y
            alpha = (tokens_seen - left_t) / (right_t - left_t)
            return left_y + alpha * (right_y - left_y)
    return parsed[-1][1]


def lr_multiplier(
    tokens_seen: int,
    *,
    total_tokens: int,
    schedule: dict[str, Any] | None,
    legacy_warmup_tokens: int,
    legacy_min_lr_ratio: float,
) -> float:
    """Evaluate the configured token-based LR multiplier.

    ``schedule=None`` preserves the bootstrap's cosine behavior exactly.
    """
    if schedule is None:
        return cosine_lr_multiplier(
            tokens_seen,
            total_tokens=total_tokens,
            warmup_tokens=legacy_warmup_tokens,
            min_lr_ratio=legacy_min_lr_ratio,
        )
    kind = str(schedule.get("type", "cosine"))
    if kind == "constant":
        return 1.0
    if kind == "cosine":
        return cosine_lr_multiplier(
            tokens_seen,
            total_tokens=total_tokens,
            warmup_tokens=int(schedule.get("warmup_tokens", 0)),
            min_lr_ratio=float(schedule.get("min_multiplier", 0.1)),
        )
    if kind == "piecewise_linear":
        return piecewise_linear_multiplier(tokens_seen, schedule["points"])
    raise ValueError(f"unknown LR schedule type {kind!r}")
