from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _validate_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B,H,T,D]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key batch and head dimensions must align")
    if query.shape[-2] != key.shape[-2]:
        raise ValueError("strict-past local attention requires equal Q/K sequence lengths")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")


def strict_past_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    window: int,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Exact O(T*W) GQA attention to the previous ``W`` memory positions.

    Query position ``t`` may read keys ``max(0,t-W) .. t-1``. It can never
    read the same-position or a future previous-pass state. Position zero has
    an empty memory set and therefore returns an exact zero vector.
    """
    _validate_qkv(query, key, value)
    if window <= 0:
        raise ValueError("window must be positive")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0,1)")

    bsz, hq, seq_len, head_dim = query.shape
    hkv = key.shape[1]
    if seq_len == 0:
        return query.clone()
    window = min(int(window), seq_len)
    groups = hq // hkv
    grouped_query = query.reshape(bsz, hkv, groups, seq_len, head_dim)

    # Left-padding by W (rather than W-1 as in ordinary inclusive causal SWA)
    # makes window t contain exactly original memory indices [t-W, ..., t-1].
    padded_key = F.pad(key, (0, 0, window, 0))
    padded_value = F.pad(value, (0, 0, window, 0))
    key_windows = (
        padded_key.unfold(dimension=-2, size=window, step=1)
        .permute(0, 1, 2, 4, 3)[:, :, :seq_len, :, :]
    )
    value_windows = (
        padded_value.unfold(dimension=-2, size=window, step=1)
        .permute(0, 1, 2, 4, 3)[:, :, :seq_len, :, :]
    )

    q_for_mm = grouped_query.permute(0, 1, 3, 2, 4)  # [B,Hkv,T,G,D]
    scores = torch.matmul(q_for_mm, key_windows.transpose(-2, -1))
    scores = scores.permute(0, 1, 3, 2, 4) / math.sqrt(head_dim)  # [B,Hkv,G,T,W]

    t = torch.arange(seq_len, device=query.device)
    j = torch.arange(window, device=query.device)
    valid = (t[:, None] - window + j[None, :]) >= 0
    scores = scores.masked_fill(~valid[None, None, None, :, :], torch.finfo(scores.dtype).min)

    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = probabilities * valid[None, None, None, :, :].to(probabilities.dtype)
    denominator = probabilities.sum(dim=-1, keepdim=True)
    probabilities = torch.where(
        denominator > 0,
        probabilities / denominator.clamp_min(torch.finfo(probabilities.dtype).tiny),
        torch.zeros_like(probabilities),
    )
    if dropout_p:
        probabilities = F.dropout(probabilities, p=dropout_p, training=training)

    probs_for_mm = probabilities.permute(0, 1, 3, 2, 4)  # [B,Hkv,T,G,W]
    output = torch.matmul(probs_for_mm, value_windows)  # [B,Hkv,T,G,D]
    output = output.permute(0, 1, 3, 2, 4).contiguous()
    return output.reshape(bsz, hq, seq_len, head_dim)
