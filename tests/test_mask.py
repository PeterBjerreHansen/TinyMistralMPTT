import torch

from tiny_mistral.attention.reference import make_allowed_mask


def test_v4452_window_semantics_allows_window_total_positions():
    T, W = 40, 32
    pos = torch.arange(T)[None, :]
    mask = make_allowed_mask(pos, pos, sliding_window=W)[0]
    q = 35
    allowed = torch.where(mask[q])[0].tolist()
    assert allowed[0] == 4  # 35 - (32 - 1)
    assert allowed[-1] == 35
    assert len(allowed) == 32
    assert not bool(mask[q, 3])


def test_causal_no_future():
    pos = torch.arange(8)[None, :]
    mask = make_allowed_mask(pos, pos, sliding_window=None)[0]
    assert bool(mask[5, 5])
    assert not bool(mask[5, 6])
