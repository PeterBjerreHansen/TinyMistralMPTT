import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant
from tiny_mistral_mptt.variants.multipass import shift_previous_hidden


def make_variant():
    torch.manual_seed(30)
    backbone = MistralForCausalLM(micro_config(), attention_backend="reference")
    return MemoryAddVariant(backbone)


def test_memory_add_one_pass_is_exact_vanilla():
    variant = make_variant().eval()
    ids = torch.tensor([[1, 4, 8, 12, 16]])
    with torch.no_grad():
        direct = variant.backbone(ids, use_cache=False).logits
        pass_one = variant.compute_passes(ids, passes=1).passes[0].logits
    torch.testing.assert_close(pass_one, direct, atol=0, rtol=0)


def test_memory_add_zero_initialization_makes_all_passes_exact_vanilla():
    variant = make_variant().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        direct = variant.backbone(ids, use_cache=False).logits
        result = variant.compute_passes(ids, passes=4)
    assert torch.count_nonzero(variant.memory_projection.weight) == 0
    for pass_result in result.passes:
        torch.testing.assert_close(pass_result.logits, direct, atol=0, rtol=0)


def test_memory_add_uses_strict_previous_position_and_zero_at_position_zero():
    variant = make_variant().eval()
    dim = variant.config.hidden_size
    embeddings = torch.randn(1, 5, dim)
    previous = torch.randn(1, 5, dim)
    with torch.no_grad():
        variant.memory_projection.weight.copy_(torch.eye(dim))
        actual = variant.feedback_inputs(embeddings, previous)
        normalized = variant.memory_norm(previous)

    torch.testing.assert_close(actual[:, 0], embeddings[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], embeddings[:, 1:] + normalized[:, :-1], atol=0, rtol=0
    )


def test_shared_shift_helper_has_exact_causal_alignment():
    previous = torch.arange(1 * 4 * 3, dtype=torch.float32).view(1, 4, 3)
    shifted = shift_previous_hidden(previous)
    torch.testing.assert_close(shifted[:, 0], torch.zeros_like(shifted[:, 0]))
    torch.testing.assert_close(shifted[:, 1:], previous[:, :-1])


def test_memory_add_zero_previous_state_is_exact_vanilla_even_after_reader_changes():
    variant = make_variant().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        variant.memory_projection.weight.normal_(mean=0.0, std=0.1)
        embeddings = variant.backbone.model.embed_tokens(ids)
        previous = torch.zeros_like(embeddings)
        expected = variant.backbone.model(
            inputs_embeds=embeddings, use_cache=False
        ).last_hidden_state
        actual = variant._run_feedback_hidden(ids, embeddings, previous)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_memory_add_phase_a_trains_only_added_parameters_and_projection_has_gradient():
    variant = make_variant()
    configure_phase(variant, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = variant.compute_loss(
        ids, phase="A", passes=2, loss_weights=[0.0, 1.0]
    )
    output.loss.backward()

    projection_grad = variant.memory_projection.weight.grad
    assert projection_grad is not None
    assert torch.isfinite(projection_grad).all()
    assert torch.count_nonzero(projection_grad) > 0

    added_ids = {id(parameter) for parameter in variant.added_parameters()}
    for parameter in variant.parameters():
        if id(parameter) not in added_ids:
            assert parameter.grad is None


def test_memory_add_phase_b_later_pass_loss_backpropagates_to_backbone():
    variant = make_variant()
    with torch.no_grad():
        variant.memory_projection.weight.normal_(mean=0.0, std=0.01)
    configure_phase(variant, "B")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = variant.compute_loss(
        ids, phase="B", passes=2, loss_weights=[0.0, 1.0]
    )
    output.loss.backward()
    grad = variant.backbone.model.embed_tokens.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_memory_add_construction_does_not_advance_global_rng():
    torch.manual_seed(12345)
    backbone = MistralForCausalLM(micro_config(), attention_backend="reference")
    state_before = torch.get_rng_state().clone()
    MemoryAddVariant(backbone)
    state_after = torch.get_rng_state()
    torch.testing.assert_close(state_after, state_before, atol=0, rtol=0)


def test_memory_add_factory_and_config_registration():
    from tiny_mistral_mptt.config import ExperimentConfig
    from tiny_mistral_mptt.model_factory import build_variant

    cfg = ExperimentConfig.from_dict(
        {
            "variant": "memory_add",
            "phase": "A",
            "pass_schedule": [{"probabilities": {2: 1.0}}],
            "eval_passes": 2,
        }
    )
    assert cfg.variant == "memory_add"

    torch.manual_seed(44)
    backbone = MistralForCausalLM(
        micro_config(), attention_backend="reference"
    ).to(dtype=torch.float64)
    variant = build_variant("memory_add", backbone)
    assert isinstance(variant, MemoryAddVariant)
    assert variant.memory_projection.weight.dtype == torch.float64
    assert variant.memory_norm.weight.dtype == torch.float64


def test_memory_add_state_dict_roundtrip_includes_added_parameters():
    source = make_variant()
    target = make_variant()
    with torch.no_grad():
        source.memory_projection.weight.normal_(mean=0.0, std=0.03)
        source.memory_norm.weight.fill_(0.75)
    target.load_state_dict(source.state_dict(), strict=True)
    torch.testing.assert_close(
        target.memory_projection.weight, source.memory_projection.weight
    )
    torch.testing.assert_close(target.memory_norm.weight, source.memory_norm.weight)
