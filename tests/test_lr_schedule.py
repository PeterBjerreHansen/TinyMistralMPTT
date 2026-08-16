import pytest

from tiny_mistral_mptt.training.schedule import lr_multiplier, piecewise_linear_multiplier


def test_piecewise_linear_schedule_interpolates_and_clamps():
    points = [[0, 0.2], [100, 1.0], [200, 0.5]]
    assert piecewise_linear_multiplier(0, points) == pytest.approx(0.2)
    assert piecewise_linear_multiplier(50, points) == pytest.approx(0.6)
    assert piecewise_linear_multiplier(150, points) == pytest.approx(0.75)
    assert piecewise_linear_multiplier(999, points) == pytest.approx(0.5)


def test_constant_lr_schedule_is_one():
    assert lr_multiplier(
        123,
        total_tokens=1000,
        schedule={"type": "constant"},
        legacy_warmup_tokens=0,
        legacy_min_lr_ratio=0.1,
    ) == 1.0
