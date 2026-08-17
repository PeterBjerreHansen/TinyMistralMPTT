import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.vanilla import VanillaVariant


def test_vanilla_wrapper_is_exact_forward_identity():
    backbone = MistralForCausalLM(micro_config(), attention_backend="reference").eval()
    wrapped = VanillaVariant(backbone).eval()
    ids = torch.tensor([[1, 4, 8, 12, 16]])
    with torch.no_grad():
        direct = backbone(ids, use_cache=False).logits
        through_wrapper = wrapped(ids, use_cache=False).logits
    torch.testing.assert_close(through_wrapper, direct, atol=0, rtol=0)


def test_vanilla_phase_a_is_explicit_noop_and_phase_b_unfreezes():
    wrapped = VanillaVariant(MistralForCausalLM(micro_config(), attention_backend="reference"))
    assert not tuple(wrapped.added_parameters())
    assert configure_phase(wrapped, "A") == 0
    assert not any(parameter.requires_grad for parameter in wrapped.parameters())
    trainable = configure_phase(wrapped, "B")
    assert trainable == sum(parameter.numel() for parameter in wrapped.parameters())


def test_vanilla_rejects_multiple_passes():
    wrapped = VanillaVariant(MistralForCausalLM(micro_config(), attention_backend="reference"))
    with pytest.raises(ValueError, match="one pass"):
        wrapped.compute_loss(torch.tensor([[1, 2, 3]]), passes=2)
