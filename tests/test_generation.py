import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM


def test_greedy_generate_shape_and_prefix_preserved():
    cfg = micro_config(eos_token_id=9999)
    model = MistralForCausalLM(cfg, attention_backend="reference").eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    out = model.generate(prefix, 5, temperature=0.0)
    assert out.shape == (1, 9)
    assert torch.equal(out[:, :4], prefix)
