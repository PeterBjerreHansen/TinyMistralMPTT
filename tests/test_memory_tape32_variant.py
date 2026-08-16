import torch
from conftest import micro_config

from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.memory_tape32 import MemoryTape32Variant


def make_variant():
    torch.manual_seed(20)
    backbone = MistralForCausalLM(micro_config(), attention_backend="reference")
    return MemoryTape32Variant(backbone, memory_window=4, initialization_seed=321)


def test_memory_tape_one_pass_is_exact_vanilla():
    variant = make_variant().eval()
    ids = torch.tensor([[1, 4, 8, 12, 16]])
    with torch.no_grad():
        direct = variant.backbone(ids, use_cache=False).logits
        pass_one = variant.compute_passes(ids, passes=1).passes[0].logits
    torch.testing.assert_close(pass_one, direct, atol=0, rtol=0)


def test_memory_tape_manual_decoder_is_vanilla_when_memory_residual_is_zero():
    variant = make_variant().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    embeddings = variant.backbone.model.embed_tokens(ids)
    previous = torch.randn_like(embeddings)
    with torch.no_grad():
        for reader in variant.memory_readers:
            reader.o_proj.weight.zero_()
        expected = variant.backbone.model(
            inputs_embeds=embeddings, use_cache=False
        ).last_hidden_state
        actual = variant._run_feedback_hidden(ids, embeddings, previous)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_memory_tape_phase_a_trains_only_memory_readers_and_has_gradient():
    variant = make_variant()
    configure_phase(variant, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = variant.compute_loss(ids, phase="A", passes=2, loss_weights=[0.0, 1.0])
    output.loss.backward()

    added_ids = {id(parameter) for parameter in variant.added_parameters()}
    assert any(parameter.grad is not None for parameter in variant.added_parameters())
    for parameter in variant.parameters():
        if id(parameter) not in added_ids:
            assert parameter.grad is None


def test_memory_tape_phase_b_later_pass_loss_backpropagates_through_previous_pass():
    variant = make_variant()
    configure_phase(variant, "B")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = variant.compute_loss(ids, phase="B", passes=2, loss_weights=[0.0, 1.0])
    output.loss.backward()
    assert variant.backbone.model.embed_tokens.weight.grad is not None
    assert torch.isfinite(variant.backbone.model.embed_tokens.weight.grad).all()
