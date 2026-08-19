import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.variants.memory_add_sparse_tape import MemoryAddSparseTapeVariant
from tiny_mistral_mptt.variants.sparse_memory_tape import SparseMemoryTapeVariant


def backbone():
    torch.manual_seed(42)
    return MistralForCausalLM(micro_config(), attention_backend="reference")


def test_factory_builds_sparse_variant_with_write_contract():
    model = build_variant(
        "sparse_memory_tape",
        backbone(),
        memory_window=5,
        memory_write_stride=4,
        architecture_seed=99,
    )
    assert isinstance(model, SparseMemoryTapeVariant)
    assert model.memory_window == 5
    assert model.memory_write_stride == 4


def test_factory_builds_token_triggered_hybrid():
    model = build_variant(
        "memory_add_sparse_tape",
        backbone(),
        memory_window=6,
        memory_write_mode="token",
        memory_token_id=7,
        architecture_seed=99,
    )
    assert isinstance(model, MemoryAddSparseTapeVariant)
    assert model.memory_write_mode == "token"
    assert model.memory_token_id == 7


def test_sparse_writer_is_identity_and_construction_does_not_advance_global_rng():
    bb = backbone()
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    model = build_variant(
        "sparse_memory_tape",
        bb,
        memory_write_stride=1,
        architecture_seed=99,
    )
    after = torch.random.get_rng_state()
    # Reader construction is forked and identity writer consumes no RNG.
    torch.testing.assert_close(after, before, atol=0, rtol=0)
    torch.testing.assert_close(
        model.writer.proj.weight,
        torch.eye(model.config.hidden_size),
        atol=0,
        rtol=0,
    )


def test_sparse_and_hybrid_state_dict_round_trip_strictly():
    for name in ("sparse_memory_tape", "memory_add_sparse_tape"):
        a = build_variant(name, backbone(), memory_write_stride=4, architecture_seed=17)
        b = build_variant(name, backbone(), memory_write_stride=4, architecture_seed=18)
        b.load_state_dict(a.state_dict(), strict=True)
        assert set(a.state_dict()) == set(b.state_dict())
        for key, value in a.state_dict().items():
            torch.testing.assert_close(b.state_dict()[key], value, atol=0, rtol=0)
