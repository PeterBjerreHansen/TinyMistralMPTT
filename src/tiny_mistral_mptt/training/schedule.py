from __future__ import annotations

import math


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
