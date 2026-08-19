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


def test_bfloat16_autocast_requires_fp32_parameter_storage():
    cfg = _config(dtype="float32", autocast_dtype="bfloat16")
    assert cfg.autocast_dtype == "bfloat16"

    with pytest.raises(ValueError, match="requires dtype=float32"):
        _config(dtype="bfloat16", autocast_dtype="bfloat16")


def test_unvalidated_autocast_dtype_is_rejected():
    with pytest.raises(ValueError, match="autocast_dtype"):
        _config(autocast_dtype="float16")



def test_sparse_memory_config_requires_coherent_write_policy():
    cfg = _config(
        variant="sparse_memory_tape",
        memory_write_mode="periodic",
        memory_write_stride=4,
    )
    assert cfg.memory_write_stride == 4

    cfg = _config(
        variant="memory_add_sparse_tape",
        memory_write_mode="token",
        memory_write_stride=8,
        memory_token_id=7,
    )
    assert cfg.memory_token_id == 7

    with pytest.raises(ValueError, match="explicitly set memory_write_mode"):
        _config(variant="sparse_memory_tape")
    with pytest.raises(ValueError, match="explicitly set memory_write_stride"):
        _config(variant="sparse_memory_tape", memory_write_mode="periodic")

    with pytest.raises(ValueError, match="requires memory_token_id"):
        _config(
            variant="sparse_memory_tape",
            memory_write_mode="token",
            memory_write_stride=8,
        )
    with pytest.raises(ValueError, match="must not set memory_token_id"):
        _config(
            variant="sparse_memory_tape",
            memory_write_mode="periodic",
            memory_write_stride=8,
            memory_token_id=7,
        )


def test_sparse_write_fields_cannot_silently_change_other_variants():
    with pytest.raises(ValueError, match=r"memory_write_\* fields"):
        _config(variant="memory_add", memory_write_stride=4)
