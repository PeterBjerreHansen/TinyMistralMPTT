from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand GQA K/V heads to query-head count, matching HF Mistral semantics."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def make_allowed_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    sliding_window: int | None,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [B, Q, K] boolean allowed-attention mask.

    v4.45.2 Mistral convention: key is allowed when it is not in the future and
    q_pos - k_pos < sliding_window. Therefore window=32 permits the current
    position plus 31 previous positions (32 visible keys after warm-up).
    """
    if query_positions.ndim != 2 or key_positions.ndim != 2:
        raise ValueError("query_positions/key_positions must be [B, T]")
    q = query_positions[:, :, None]
    k = key_positions[:, None, :]
    allowed = k <= q
    if sliding_window is not None:
        allowed = allowed & ((q - k) < sliding_window)
    if key_padding_mask is not None:
        if key_padding_mask.shape != key_positions.shape:
            raise ValueError(
                f"key_padding_mask shape {tuple(key_padding_mask.shape)} does not match "
                f"key positions {tuple(key_positions.shape)}"
            )
        allowed = allowed & key_padding_mask[:, None, :].to(torch.bool)
    return allowed


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    sliding_window: int | None,
    key_padding_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Obvious O(QK) correctness implementation; not intended for long training."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B, H, T, D]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key batch and head dimensions must align")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")

    n_rep = query.shape[1] // key.shape[1]
    key = repeat_kv(key, n_rep)
    value = repeat_kv(value, n_rep)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    allowed = make_allowed_mask(
        query_positions,
        key_positions,
        sliding_window=sliding_window,
        key_padding_mask=key_padding_mask,
    )
    scores = scores.masked_fill(~allowed[:, None, :, :], torch.finfo(scores.dtype).min)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout_p:
        probs = F.dropout(probs, p=dropout_p, training=training)
    return torch.matmul(probs, value)
