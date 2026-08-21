import pytest

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.variants.tape import TapeVariant
from tiny_mistral_mptt.variants.tape_add_hybrid import TapeAddHybridVariant
from tiny_mistral_mptt.variants.tape_recirculation_hybrid import (
    TapeRecirculationHybridVariant,
)
from tiny_mistral_mptt.variants.recirculation import RecirculationVariant


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


def test_factory_selects_only_requested_memory_layers_and_defaults_to_rope():
    multi_layer = MistralForCausalLM(
        micro_config(num_hidden_layers=3), attention_backend="reference"
    )
    model = build_variant(
        "tape",
        multi_layer,
        memory_write_mode="periodic",
        memory_write_stride=8,
        memory_layers=[0, 2],
    )
    assert model.memory_layers == (0, 2)
    assert list(model.memory_readers) == ["0", "2"]
    assert all(reader.position_encoding == "rope" for reader in model.memory_readers.values())

    with pytest.raises(ValueError, match="memory_layers"):
        build_variant(
            "tape",
            MistralForCausalLM(
                micro_config(num_hidden_layers=2), attention_backend="reference"
            ),
            memory_write_mode="periodic",
            memory_write_stride=8,
            memory_layers=[2],
        )


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


def test_factory_builds_recirculation_with_explicit_layer_contract():
    two_layer_backbone = MistralForCausalLM(
        micro_config(num_hidden_layers=2), attention_backend="reference"
    )
    model = build_variant(
        "recirculation",
        two_layer_backbone,
        recirculation_source_layer=1,
        recirculation_destination_layer=0,
    )
    assert isinstance(model, RecirculationVariant)

    adaptive = build_variant(
        "recirculation",
        MistralForCausalLM(
            micro_config(num_hidden_layers=2), attention_backend="reference"
        ),
        recirculation_source_layer=1,
        recirculation_destination_layer=0,
        recirculation_mode="adaptive",
    )
    assert isinstance(adaptive, RecirculationVariant)
    assert adaptive.mode == "adaptive"
    assert list(adaptive.added_parameters())


def test_factory_builds_adaptive_tape_recirculation_hybrid():
    model = build_variant(
        "tape_recirculation_hybrid",
        MistralForCausalLM(
            micro_config(num_hidden_layers=3), attention_backend="reference"
        ),
        memory_write_mode="periodic",
        memory_write_stride=8,
        memory_layers=[1],
        recirculation_source_layer=2,
        recirculation_destination_layer=0,
        recirculation_mode="adaptive",
    )
    assert isinstance(model, TapeRecirculationHybridVariant)
    assert model.mode == "adaptive"
    assert model.memory_layers == (1,)
