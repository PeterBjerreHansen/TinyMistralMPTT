from tiny_mistral.config import MistralConfig, tiny_mistral_248m_config


def test_tiny_mistral_config_exact_fields():
    cfg = tiny_mistral_248m_config()
    assert cfg.vocab_size == 32005
    assert cfg.hidden_size == 1024
    assert cfg.intermediate_size == 4096
    assert cfg.num_hidden_layers == 12
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 32
    assert cfg.sliding_window == 32
    assert cfg.transformers_version == "4.45.2"


def test_config_roundtrip(tmp_path):
    cfg = tiny_mistral_248m_config()
    path = tmp_path / "config.json"
    cfg.to_json_file(path)
    loaded = MistralConfig.from_json_file(path)
    assert loaded == cfg


def test_checked_in_checkpoint_config_snapshot_matches_factory():
    from pathlib import Path
    path = Path(__file__).parents[1] / "configs" / "tinymistral_248m_base.json"
    assert MistralConfig.from_json_file(path) == tiny_mistral_248m_config()
