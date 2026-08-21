import pytest

from tiny_mistral_mptt.config import ExperimentConfig


def test_interruptible_checkpoint_defaults_are_safe():
    cfg = ExperimentConfig()
    assert cfg.checkpoint_keep_last == 2
    assert cfg.checkpoint_every_seconds == 0.0


def test_local_checkpoint_retention_can_drop_to_one():
    cfg = ExperimentConfig.from_dict({"checkpoint_keep_last": 1})
    assert cfg.checkpoint_keep_last == 1


def test_checkpoint_retention_cannot_drop_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        ExperimentConfig.from_dict({"checkpoint_keep_last": 0})


def test_snapshot_thresholds_are_normalized():
    cfg = ExperimentConfig.from_dict(
        {"max_unique_tokens": 100, "snapshot_at_tokens": [100, 25, 25, 50]}
    )
    assert cfg.snapshot_at_tokens == [25, 50, 100]
