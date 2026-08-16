from __future__ import annotations

import pytest
import torch

from conftest import micro_config
from tiny_mistral.attention.local import local_window_attention
from tiny_mistral.attention.reference import reference_attention
from tiny_mistral.modeling import MistralAttention, MistralForCausalLM


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 5, 17, 33])
def test_local_kernel_matches_reference(seq_len: int):
    batch, hq, hkv, dim = 2, 4, 2, 8
    q = torch.randn(batch, hq, seq_len, dim)
    k = torch.randn(batch, hkv, seq_len, dim)
    v = torch.randn(batch, hkv, seq_len, dim)
    pos = torch.arange(seq_len)[None, :].expand(batch, -1)

    expected = reference_attention(
        q,
        k,
        v,
        query_positions=pos,
        key_positions=pos,
        sliding_window=4,
    )
    actual = local_window_attention(q, k, v, sliding_window=4)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_local_kernel_matches_reference_without_bounded_window():
    batch, hq, hkv, seq_len, dim = 1, 6, 2, 9, 4
    q = torch.randn(batch, hq, seq_len, dim)
    k = torch.randn(batch, hkv, seq_len, dim)
    v = torch.randn(batch, hkv, seq_len, dim)
    pos = torch.arange(seq_len)[None, :]
    expected = reference_attention(
        q,
        k,
        v,
        query_positions=pos,
        key_positions=pos,
        sliding_window=None,
    )
    actual = local_window_attention(q, k, v, sliding_window=None)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_local_kernel_backward_matches_reference():
    batch, hq, hkv, seq_len, dim = 1, 4, 2, 11, 8
    q0 = torch.randn(batch, hq, seq_len, dim)
    k0 = torch.randn(batch, hkv, seq_len, dim)
    v0 = torch.randn(batch, hkv, seq_len, dim)
    pos = torch.arange(seq_len)[None, :]

    q1, k1, v1 = (x.clone().requires_grad_(True) for x in (q0, k0, v0))
    q2, k2, v2 = (x.clone().requires_grad_(True) for x in (q0, k0, v0))

    a = reference_attention(
        q1,
        k1,
        v1,
        query_positions=pos,
        key_positions=pos,
        sliding_window=4,
    )
    b = local_window_attention(q2, k2, v2, sliding_window=4)
    weight = torch.randn_like(a)
    (a * weight).sum().backward()
    (b * weight).sum().backward()

    for got, expected in ((q2.grad, q1.grad), (k2.grad, k1.grad), (v2.grad, v1.grad)):
        assert got is not None and expected is not None
        torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-5)


def test_attention_module_local_matches_reference():
    cfg = micro_config(sliding_window=4)
    ref = MistralAttention(cfg, 0, attention_backend="reference")
    local = MistralAttention(cfg, 0, attention_backend="local")
    local.load_state_dict(ref.state_dict())
    ref.eval(); local.eval()
    x = torch.randn(2, 17, cfg.hidden_size)
    pos = torch.arange(17)[None, :].expand(2, -1)
    with torch.no_grad():
        a, _ = ref(x, attention_mask=None, position_ids=pos, use_cache=False)
        b, _ = local(x, attention_mask=None, position_ids=pos, use_cache=False)
    torch.testing.assert_close(b, a, atol=2e-5, rtol=2e-5)


def test_full_model_local_matches_reference():
    cfg = micro_config(sliding_window=4)
    ref = MistralForCausalLM(cfg, attention_backend="reference").eval()
    local = MistralForCausalLM(cfg, attention_backend="local").eval()
    local.load_state_dict(ref.state_dict())
    ids = torch.randint(0, cfg.vocab_size, (1, 23))
    with torch.no_grad():
        a = ref(ids, use_cache=False).logits
        b = local(ids, use_cache=False).logits
    torch.testing.assert_close(b, a, atol=3e-5, rtol=3e-5)


def test_local_backend_falls_back_for_padding():
    cfg = micro_config(sliding_window=4)
    ref = MistralAttention(cfg, 0, attention_backend="reference")
    local = MistralAttention(cfg, 0, attention_backend="local")
    local.load_state_dict(ref.state_dict())
    x = torch.randn(1, 8, cfg.hidden_size)
    pos = torch.arange(8)[None, :]
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    with torch.no_grad():
        a, _ = ref(x, attention_mask=mask, position_ids=pos)
        b, _ = local(x, attention_mask=mask, position_ids=pos)
    torch.testing.assert_close(b, a)
