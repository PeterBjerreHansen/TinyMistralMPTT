from types import SimpleNamespace

import torch

from tiny_mistral_mptt.evaluation.lm_eval_adapter import score_token_continuation


class NextIdModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, ids, use_cache=False):
        batch, length = ids.shape
        logits = torch.full((batch, length, self.vocab_size), -20.0)
        next_ids = (ids + 1) % self.vocab_size
        logits.scatter_(-1, next_ids.unsqueeze(-1), 20.0)
        return SimpleNamespace(logits=logits)


def test_token_continuation_scores_only_continuation_and_greedy_contract():
    model = NextIdModel()
    score, greedy = score_token_continuation(
        model,
        device="cpu",
        max_length=16,
        context_enc=[10, 11],
        continuation_enc=[12, 13],
    )
    assert greedy is True
    assert score > -1e-5

    _score, greedy = score_token_continuation(
        model,
        device="cpu",
        max_length=16,
        context_enc=[10, 11],
        continuation_enc=[12, 14],
    )
    assert greedy is False


def test_token_continuation_left_truncation_keeps_requested_targets():
    model = NextIdModel()
    score, greedy = score_token_continuation(
        model,
        device="cpu",
        max_length=3,
        context_enc=[7, 8, 9],
        continuation_enc=[10, 11],
    )
    assert greedy is True
    assert score > -1e-5
