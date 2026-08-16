import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM


def test_generate_projects_only_last_prefill_hidden_state():
    cfg = micro_config(eos_token_id=9999)
    model = MistralForCausalLM(cfg, attention_backend="reference").eval()
    seen_lengths = []
    handle = model.lm_head.register_forward_hook(
        lambda _module, inputs, _output: seen_lengths.append(inputs[0].shape[1])
    )
    try:
        model.generate(torch.tensor([[1, 2, 3, 4]]), 2)
    finally:
        handle.remove()
    assert seen_lengths == [1, 1]


def test_generate_rejects_batch_and_invalid_top_k():
    model = MistralForCausalLM(micro_config(eos_token_id=9999), attention_backend="reference").eval()
    with pytest.raises(ValueError, match="batch size 1"):
        model.generate(torch.tensor([[1, 2], [3, 4]]), 1)
    with pytest.raises(ValueError, match="top_k"):
        model.generate(torch.tensor([[1, 2]]), 1, top_k=0)
