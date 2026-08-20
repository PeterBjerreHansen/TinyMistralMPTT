import pytest

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.variants.tape import TapeVariant
from tiny_mistral_mptt.variants.tape_add_hybrid import TapeAddHybridVariant


def backbone():
    return MistralForCausalLM(micro_config(num_hidden_layers=1), attention_backend="reference")


def test_factory_exposes_only_clean_tape_names_and_policies():
    dense = build_variant("tape", backbone(), memory_write_mode="dense")
    assert isinstance(dense, TapeVariant)
    assert dense.memory_write_mode == "dense"

    hybrid = build_variant(
        "tape_add_hybrid",
        backbone(),
        memory_write_mode="memory_token",
        memory_write_stride=8,
        memory_token_visibility="write_only",
    )
    assert isinstance(hybrid, TapeAddHybridVariant)


@pytest.mark.parametrize(
    "legacy_name",
    ["memory_tape32", "dense_memory_tape", "sparse_memory_tape", "memory_add_sparse_tape"],
)
def test_factory_rejects_removed_tape_aliases(legacy_name):
    with pytest.raises(ValueError, match="unknown variant"):
        build_variant(legacy_name, backbone())


def test_factory_requires_memory_token_visibility_explicitly():
    with pytest.raises(ValueError, match="memory_token_visibility"):
        build_variant(
            "tape",
            backbone(),
            memory_write_mode="memory_token",
            memory_write_stride=8,
        )
