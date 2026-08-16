import torch

from tiny_mistral_mptt.training.loss import normalize_pass_weights


def test_pass_weights_are_right_aligned_and_normalized():
    weights = normalize_pass_weights(
        [0.05, 0.20, 0.75],
        2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(weights, torch.tensor([0.20, 0.75]) / 0.95)


def test_short_pass_weight_vector_is_left_zero_padded():
    weights = normalize_pass_weights(
        [1.0],
        3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(weights, torch.tensor([0.0, 0.0, 1.0]))


def test_default_weights_are_uniform():
    weights = normalize_pass_weights(
        None, 3, device=torch.device("cpu"), dtype=torch.float32
    )
    torch.testing.assert_close(weights, torch.full((3,), 1 / 3))
