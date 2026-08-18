import pytest

from tiny_mistral_mptt.config import ExperimentConfig


def _config(**overrides):
    values = {
        "variant": "fbt",
        "model_dir": "unused",
        "data_dir": "unused",
        "output_dir": "unused",
        "pass_schedule": [{"probabilities": {2: 0.5, 3: 0.5}}],
        "eval_every_tokens": 0,
        "checkpoint_every_tokens": 0,
    }
    values.update(overrides)
    cfg = ExperimentConfig(**values)
    cfg.validate()
    return cfg


def test_k_specific_pass_weights_are_selected_exactly():
    cfg = _config(
        pass_loss_weights_by_k={
            2: [0.25, 0.75],
            3: [0.05, 0.20, 0.75],
        }
    )

    assert cfg.loss_weights_for_passes(2) == [0.25, 0.75]
    assert cfg.loss_weights_for_passes(3) == [0.05, 0.20, 0.75]


def test_k_specific_weights_require_exact_schedule_coverage():
    with pytest.raises(ValueError, match="exactly match"):
        _config(pass_loss_weights_by_k={2: [0.25, 0.75]})

    with pytest.raises(ValueError, match="exactly match"):
        _config(pass_loss_weights_by_k={2: [0.25, 0.75], 3: [0.05, 0.20, 0.75], 4: [1.0]})


def test_global_and_k_specific_weights_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(
            pass_loss_weights=[0.25, 0.75],
            pass_loss_weights_by_k={
                2: [0.25, 0.75],
                3: [0.05, 0.20, 0.75],
            },
        )


def test_yaml_style_string_k_keys_are_canonicalized():
    cfg = _config(
        pass_loss_weights_by_k={
            "2": [0.25, 0.75],
            "3": [0.05, 0.20, 0.75],
        }
    )

    assert cfg.pass_loss_weights_by_k == {
        2: [0.25, 0.75],
        3: [0.05, 0.20, 0.75],
    }
