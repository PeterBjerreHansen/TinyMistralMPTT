from conftest import micro_config
from tiny_mistral.config import tiny_mistral_248m_config
from tiny_mistral.loading import EXPECTED_PARAMETER_COUNT, expected_state_metadata
from tiny_mistral.modeling import MistralForCausalLM


def test_parameter_key_hierarchy_matches_hf_names():
    model = MistralForCausalLM(micro_config(), attention_backend="reference")
    keys = set(model.state_dict())
    assert "model.embed_tokens.weight" in keys
    assert "model.layers.0.self_attn.q_proj.weight" in keys
    assert "model.layers.0.self_attn.k_proj.weight" in keys
    assert "model.layers.0.self_attn.v_proj.weight" in keys
    assert "model.layers.0.self_attn.o_proj.weight" in keys
    assert "model.layers.0.mlp.gate_proj.weight" in keys
    assert "model.layers.0.input_layernorm.weight" in keys
    assert "model.norm.weight" in keys
    assert "lm_head.weight" in keys
    assert not any("rotary_emb.inv_freq" in k for k in keys)


def test_exact_tinymistral_parameter_count_on_meta():
    metadata = expected_state_metadata(tiny_mistral_248m_config())
    count = 0
    for shape in metadata.values():
        n = 1
        for d in shape:
            n *= d
        count += n
    assert count == EXPECTED_PARAMETER_COUNT == 248_024_064
