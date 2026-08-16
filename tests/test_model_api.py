import warnings

import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM


def test_input_ids_and_inputs_embeds_are_identical():
    cfg = micro_config()
    model = MistralForCausalLM(cfg, attention_backend="reference").eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 11))
    embeds = model.get_input_embeddings()(ids)
    with torch.no_grad():
        a = model(ids, use_cache=False).logits
        b = model(inputs_embeds=embeds, use_cache=False).logits
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_full_model_flex_matches_reference():
    cfg = micro_config(sliding_window=4)
    ref = MistralForCausalLM(cfg, attention_backend="reference").eval()
    flex = MistralForCausalLM(
        cfg,
        attention_backend="flex",
        compile_flex=False,
        flex_block_size=16,
    ).eval()
    flex.load_state_dict(ref.state_dict())
    ids = torch.randint(0, cfg.vocab_size, (1, 17))
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = ref(ids, use_cache=False).logits
        b = flex(ids, use_cache=False).logits
    torch.testing.assert_close(b, a, atol=3e-5, rtol=3e-5)


def test_hidden_state_contract():
    cfg = micro_config(num_hidden_layers=3)
    model = MistralForCausalLM(cfg, attention_backend="reference").eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 7))
    with torch.no_grad():
        out = model(ids, use_cache=False, output_hidden_states=True)
    assert out.hidden_states is not None
    assert len(out.hidden_states) == cfg.num_hidden_layers + 1
    assert out.hidden_states[-1].shape == (1, 7, cfg.hidden_size)
