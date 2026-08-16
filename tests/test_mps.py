from __future__ import annotations

import pytest
import torch

from conftest import micro_config
from tiny_mistral.device import mps_available
from tiny_mistral.modeling import MistralForCausalLM


@pytest.mark.skipif(not mps_available(), reason="MPS hardware/runtime unavailable")
def test_mps_local_forward_backward():
    device = torch.device("mps")
    cfg = micro_config(sliding_window=4)
    model = MistralForCausalLM(cfg, attention_backend="local").to(device=device, dtype=torch.float16).train()
    ids = torch.randint(0, cfg.vocab_size, (1, 17), device=device)
    out = model(ids, labels=ids, use_cache=False)
    assert out.loss is not None and bool(torch.isfinite(out.loss).item())
    out.loss.backward()
    assert any(p.grad is not None for p in model.parameters())
