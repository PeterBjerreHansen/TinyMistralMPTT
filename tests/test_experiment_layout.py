from pathlib import Path

import yaml

from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.config import load_data_config


ROOT = Path(__file__).resolve().parents[1]
LINEAGE = ROOT / "experiments" / "stage2_cleanroom_v1"


def _active_config_paths() -> list[Path]:
    paths = list((ROOT / "configs" / "substrate").glob("**/*.yaml"))
    paths.extend((ROOT / "configs" / "smoke").glob("*.yaml"))
    paths.extend((ROOT / "configs" / "stage2").glob("*.yaml"))
    paths.extend(
        path
        for path in (LINEAGE / "configs").glob("**/*.yaml")
        if path.parent.name != "data"
    )
    return sorted(paths)


def test_locked_layout_is_explicit():
    assert not (ROOT / "configs" / "stage1").exists()
    assert not (ROOT / "configs" / "mac").exists()
    assert not (ROOT / "experiments" / "stage1_starting_points").exists()
    assert not (ROOT / "experiments" / "stage2_training").exists()
    assert not (ROOT / "experiments" / "memory_phase_b").exists()

    assert (LINEAGE / "README.md").exists()
    assert (LINEAGE / "PROTOCOL.yaml").exists()
    assert list((ROOT / "configs" / "stage2").glob("*.yaml"))


def test_protocol_is_pinned():
    manifest = yaml.safe_load((LINEAGE / "PROTOCOL.yaml").read_text(encoding="utf-8"))
    assert manifest["version"] == 2
    assert manifest["lineage"] == "stage2_cleanroom_v1"
    assert manifest["data"]["artifact"] == "data/stage2_cleanroom_v1/sequence_512"
    assert manifest["starting_points"]["memory_add"]["checkpoint"].startswith(
        "runs/stage2_cleanroom_v1/"
    )
    assert manifest["starting_points"]["memory_tape32"]["checkpoint"].startswith(
        "runs/stage2_cleanroom_v1/"
    )
    assert manifest["locked_protocol"]["backbone_learning_rate"] == 3.0e-7
    assert manifest["locked_protocol"]["added_learning_rate"] == 1.0e-6
    assert manifest["locked_protocol"]["passes"] == 3
    assert manifest["locked_protocol"]["pass_loss_weights"] == [0.05, 0.20, 0.75]


def test_promoted_k3_configs_use_locked_weights():
    for path in (ROOT / "configs" / "stage2").glob("*.yaml"):
        config = load_experiment_config(path)
        assert config.pass_loss_weights == [0.05, 0.20, 0.75]


def test_data_recipe_parses():
    cfg = load_data_config(LINEAGE / "configs" / "data" / "artifact.yaml")
    assert cfg.output_dir == "data/stage2_cleanroom_v1/sequence_512"
    assert cfg.train_tokens == 1_048_576
    assert cfg.validation_tokens == 131_072


def test_all_active_experiment_configs_parse():
    paths = _active_config_paths()
    assert paths
    for path in paths:
        load_experiment_config(path)


def test_config_and_run_namespaces_are_current():
    roots = [ROOT / "configs", LINEAGE]
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in root.glob("**/*.yaml")
    )
    assert "runs/mac-" not in text
    assert "runs/cleanroom-v1" not in text
