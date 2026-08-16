from __future__ import annotations

from functools import lru_cache
from typing import Callable

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover - old torch import path
    create_block_mask = None
    flex_attention = None


@lru_cache(maxsize=64)
def _cached_local_block_mask(
    seq_len: int,
    sliding_window: int | None,
    device_type: str,
    device_index: int | None,
    block_size: int,
):
    if create_block_mask is None:
        raise RuntimeError("FlexAttention is unavailable; install PyTorch >= 2.5")
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)

    if sliding_window is None:
        def mask_mod(b, h, q_idx, kv_idx):
            return kv_idx <= q_idx
    else:
        window = int(sliding_window)

        def mask_mod(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & ((q_idx - kv_idx) < window)

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        BLOCK_SIZE=block_size,
        _compile=(device.type == "cuda"),
    )


@lru_cache(maxsize=4)
def _compiled_flex(dynamic: bool) -> Callable:
    if flex_attention is None:
        raise RuntimeError("FlexAttention is unavailable; install PyTorch >= 2.5")
    return torch.compile(flex_attention, dynamic=dynamic)


def flex_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sliding_window: int | None,
    compile_kernel: bool = True,
    block_size: int = 128,
) -> torch.Tensor:
    """Sparse causal local attention for full, unpadded sequence forwards.

    Query has Hq heads while K/V may have Hkv heads. FlexAttention's native GQA
    support avoids physically repeating K/V heads.
    """
    if flex_attention is None:
        raise RuntimeError("FlexAttention is unavailable; install PyTorch >= 2.5")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B, H, T, D]")
    if query.shape[-2] != key.shape[-2] or key.shape != value.shape:
        raise ValueError("flex full-sequence path requires Q/K/V to share sequence length")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")

    seq_len = query.shape[-2]
    device = query.device
    block_mask = _cached_local_block_mask(
        seq_len,
        sliding_window,
        device.type,
        device.index,
        block_size,
    )
    fn = _compiled_flex(dynamic=False) if compile_kernel else flex_attention
    return fn(
        query,
        key,
        value,
        block_mask=block_mask,
        enable_gqa=(query.shape[1] != key.shape[1]),
    )
