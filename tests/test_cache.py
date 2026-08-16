import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM


def test_incremental_cache_matches_full_forward_at_each_position():
    cfg = micro_config(num_hidden_layers=2, sliding_window=4)
    model = MistralForCausalLM(cfg, attention_backend="reference").eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 13))
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        cache = None
        pieces = []
        for t in range(ids.shape[1]):
            out = model(ids[:, t:t+1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            pieces.append(out.logits)
            assert cache is not None
            for layer_cache in cache:
                assert layer_cache.seq_len <= max(cfg.sliding_window - 1, 0)
        inc = torch.cat(pieces, dim=1)
    torch.testing.assert_close(inc, full, atol=3e-5, rtol=3e-5)
