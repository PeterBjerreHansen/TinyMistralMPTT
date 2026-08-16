import math

import torch

from tiny_mistral_mptt.attention.memory_local import strict_past_local_attention


def dense_reference(query, key, value, window):
    bsz, hq, seq_len, dim = query.shape
    hkv = key.shape[1]
    groups = hq // hkv
    key = key[:, :, None, :, :].expand(bsz, hkv, groups, seq_len, dim).reshape(bsz, hq, seq_len, dim)
    value = value[:, :, None, :, :].expand(bsz, hkv, groups, seq_len, dim).reshape(bsz, hq, seq_len, dim)
    scores = query @ key.transpose(-2, -1) / math.sqrt(dim)
    q = torch.arange(seq_len)[:, None]
    k = torch.arange(seq_len)[None, :]
    allowed = (k < q) & ((q - k) <= window)
    scores = scores.masked_fill(~allowed[None, None], torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    probs = probs * allowed[None, None]
    denom = probs.sum(-1, keepdim=True)
    probs = torch.where(denom > 0, probs / denom.clamp_min(torch.finfo(probs.dtype).tiny), torch.zeros_like(probs))
    return probs @ value


def test_strict_past_local_attention_matches_dense_gqa_reference():
    torch.manual_seed(4)
    q = torch.randn(2, 4, 9, 8)
    k = torch.randn(2, 2, 9, 8)
    v = torch.randn(2, 2, 9, 8)
    actual = strict_past_local_attention(q, k, v, window=4)
    expected = dense_reference(q, k, v, window=4)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[:, :, 0], torch.zeros_like(actual[:, :, 0]), atol=0, rtol=0)


def test_memory_attention_ignores_current_future_and_too_old_memory():
    torch.manual_seed(5)
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 1, 8, 4)
    v = torch.randn(1, 1, 8, 4)
    baseline = strict_past_local_attention(q, k, v, window=3)

    # Query 6 can read only memory positions 3,4,5.
    changed = v.clone()
    changed[:, :, 0] += 1000  # too old
    changed[:, :, 6:] -= 1000  # current/future
    perturbed = strict_past_local_attention(q, k, changed, window=3)
    torch.testing.assert_close(perturbed[:, :, 6], baseline[:, :, 6], atol=0, rtol=0)
