import json

import torch
from safetensors.torch import save_file

from conftest import micro_config
from tiny_mistral.loading import load_model, verify_checkpoint_structure
from tiny_mistral.modeling import MistralForCausalLM


def test_strict_safetensors_roundtrip(tmp_path):
    cfg = micro_config()
    cfg.to_json_file(tmp_path / "config.json")
    source = MistralForCausalLM(cfg, attention_backend="reference")
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    loaded = load_model(tmp_path, attention_backend="reference", device="cpu")
    for (ka, va), (kb, vb) in zip(source.state_dict().items(), loaded.state_dict().items()):
        assert ka == kb
        torch.testing.assert_close(va, vb)


def test_verify_structure_detects_clean_micro_checkpoint(tmp_path):
    cfg = micro_config()
    cfg.to_json_file(tmp_path / "config.json")
    source = MistralForCausalLM(cfg, attention_backend="reference")
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    result = verify_checkpoint_structure(tmp_path)
    assert result["ok"]
    assert result["missing_keys"] == []
    assert result["unexpected_keys"] == []
