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
    assert not list((ROOT / "configs" / "stage2").glob("*.yaml"))


def test_protocol_is_pinned():
    manifest = yaml.safe_load((LINEAGE / "PROTOCOL.yaml").read_text(encoding="utf-8"))
    assert manifest["version"] == 3
    assert manifest["status"] == "k_schedule_pending"
    assert manifest["lineage"] == "stage2_cleanroom_v1"
    assert manifest["data"]["artifact"] == "data/stage2_cleanroom_v1/sequence_512"
    assert manifest["starting_points"]["memory_add"]["checkpoint"].startswith(
        "runs/stage2_cleanroom_v1/"
    )
    assert manifest["starting_points"]["memory_tape32"]["checkpoint"].startswith(
        "runs/stage2_cleanroom_v1/"
    )
    assert manifest["locked_protocol"]["backbone_learning_rate"] == 1.0e-6
    assert manifest["locked_protocol"]["added_learning_rate"] == 1.0e-6
    assert manifest["locked_protocol"]["k_schedule"] == "pending"
    assert len(manifest["locked_protocol"]["k_sweep"]) == 4


def test_data_recipe_parses():
    cfg = load_data_config(LINEAGE / "configs" / "data" / "artifact.yaml")
    assert cfg.output_dir == "data/stage2_cleanroom_v1/sequence_512"
    assert cfg.model_dir == "checkpoints/TinyMistral-248M-v3"
    assert cfg.train_tokens == 1_048_576
    assert cfg.validation_tokens == 131_072


def test_all_active_experiment_configs_parse():
    paths = _active_config_paths()
    assert paths
    for path in paths:
        load_experiment_config(path)


def test_learning_rate_sweep_includes_high_lr_arm():
    expected = {0.0, 3.0e-8, 1.0e-7, 3.0e-7, 1.0e-6, 3.0e-6, 1.0e-5}
    for variant in ("memory_add", "memory_tape32"):
        paths = (LINEAGE / "configs" / "learning_rate" / variant).glob("backbone_lr_*.yaml")
        observed = {load_experiment_config(path).pretrained_learning_rate for path in paths}
        assert observed == expected


def test_vanilla_learning_rate_controls_are_present():
    paths = (LINEAGE / "configs" / "learning_rate" / "vanilla").glob("backbone_lr_*.yaml")
    observed = {load_experiment_config(path).learning_rate for path in paths}
    assert observed == {3.0e-7, 1.0e-6, 3.0e-6}


def test_selected_lr_k_sweep_configs_are_complete():
    expected = {"k2", "k2_90_k3_10", "k2_50_k3_50", "k3"}
    for variant in ("memory_add", "memory_tape32"):
        paths = {
            path.stem: path
            for path in (LINEAGE / "configs" / "k_sweep" / variant).glob("*.yaml")
        }
        assert set(paths) == expected
        for path in paths.values():
            config = load_experiment_config(path)
            assert config.pretrained_learning_rate == 1.0e-6
            assert config.added_learning_rate == 1.0e-6
            assert config.max_unique_tokens == 1_048_576


def test_config_and_run_namespaces_are_current():
    roots = [ROOT / "configs", LINEAGE]
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in root.glob("**/*.yaml")
    )
    assert "runs/mac-" not in text
    assert "runs/cleanroom-v1" not in text


def test_root_readme_keeps_stage2_protocol_pending():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "The Stage 2 protocol remains open" in readme
    assert "configs/stage2/memory_add_k3.yaml" not in readme
    assert "backbone learning rate `3e-7`" not in readme


def test_stage2_docs_do_not_claim_an_old_lock():
    lr_report = (LINEAGE / "results" / "learning_rate.md").read_text(encoding="utf-8")
    validation = (ROOT / "docs" / "VALIDATION.md").read_text(encoding="utf-8")
    assert "locked clean-room protocol remains unchanged" not in lr_report
    assert "The final K schedule and recurrent inference depth are not locked yet" in (
        LINEAGE / "README.md"
    ).read_text(encoding="utf-8")
    assert "current clean-room lineage" in validation
