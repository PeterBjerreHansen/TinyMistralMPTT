import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM


def test_forward_backward_optimizer_step_finite():
    cfg = micro_config()
    model = MistralForCausalLM(cfg, attention_backend="reference").train()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = model(ids, labels=ids, use_cache=False)
    assert out.loss is not None and torch.isfinite(out.loss)
    opt.zero_grad(set_to_none=True)
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    opt.step()
