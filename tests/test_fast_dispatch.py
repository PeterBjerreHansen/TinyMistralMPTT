import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralAttention


def test_model_level_fast_dispatch_flag_preserves_local_attention_math():
    cfg = micro_config(sliding_window=4)
    attn = MistralAttention(cfg, 0, attention_backend="local").eval()
    x = torch.randn(2, 9, cfg.hidden_size)
    pos = torch.arange(9)[None, :].expand(2, -1)
    with torch.no_grad():
        discovered, _ = attn(
            x,
            attention_mask=None,
            position_ids=pos,
            use_cache=False,
            fast_attention_compatible=None,
        )
        precomputed, _ = attn(
            x,
            attention_mask=None,
            position_ids=pos,
            use_cache=False,
            fast_attention_compatible=True,
        )
    torch.testing.assert_close(precomputed, discovered, atol=0, rtol=0)
