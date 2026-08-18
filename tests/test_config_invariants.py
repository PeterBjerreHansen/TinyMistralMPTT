from tiny_mistral_mptt.config import ExperimentConfig
from tiny_mistral_mptt.config_invariants import (
    differing_invariant_fields,
    execution_invariant_view,
)


def _config(**overrides):
    values = {
        "variant": "memory_add",
        "phase": "B",
        "model_dir": "model",
        "data_dir": "data",
        "output_dir": "runs/a",
        "device": "mps",
        "dtype": "float32",
        "autocast_dtype": None,
        "attention_backend": "auto",
        "pass_schedule": [{"probabilities": {2: 1.0}}],
    }
    values.update(overrides)
    config = ExperimentConfig(**values)
    config.validate()
    return config


def test_execution_invariant_view_includes_precision_and_trajectory_fields():
    view = execution_invariant_view(_config())
    assert view["variant"] == "memory_add"
    assert view["phase"] == "B"
    assert view["dtype"] == "float32"
    assert view["autocast_dtype"] is None
    assert "batch_size" in view
    assert "max_unique_tokens" in view
    assert "lr_schedule" in view


def test_differing_invariant_fields_are_explicit():
    first = execution_invariant_view(_config())
    second = execution_invariant_view(_config(batch_size=2, grad_clip=0.5))
    assert differing_invariant_fields(first, second) == ["batch_size", "grad_clip"]
