from pathlib import Path

import yaml

from tiny_mistral_mptt.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def _experiment_config_paths() -> list[Path]:
    paths = [
        ROOT / "configs" / "substrate" / "mac" / "vanilla.yaml",
        ROOT / "configs" / "substrate" / "gpu" / "vanilla.yaml",
        ROOT / "configs" / "smoke" / "vanilla.yaml",
        ROOT / "configs" / "stage1" / "mac" / "memory_add_wiring.yaml",
        ROOT / "configs" / "stage1" / "mac" / "memory_tape32_wiring.yaml",
        ROOT / "experiments" / "stage1_starting_points" / "memory_add_wired_checkpoint.yaml",
        ROOT / "experiments" / "stage1_starting_points" / "memory_tape32_wired_checkpoint.yaml",
    ]
    paths.extend(
        path
        for path in (
            ROOT
            / "experiments"
            / "stage1_starting_points"
            / "fbt_retrofit"
            / "configs"
        ).glob("*.yaml")
        if not path.name.endswith(".historical.yaml")
    )
    paths.extend(
        (
            ROOT
            / "experiments"
            / "stage2_training"
            / "protocol_development"
            / "learning_rate"
            / "configs"
        ).glob("**/*.yaml")
    )
    paths.extend(
        (
            ROOT
            / "experiments"
            / "stage2_training"
            / "protocol_development"
            / "pass_depth"
            / "configs"
        ).glob("*.yaml")
    )
    return sorted(set(paths))


def test_research_stage_layout_is_explicit():
    assert not (ROOT / "configs" / "mac").exists()
    assert not (ROOT / "experiments" / "memory_phase_b").exists()
    assert not (ROOT / "docs" / "EXPERIMENT.md").exists()

    stage1 = ROOT / "configs" / "stage1" / "mac"
    assert {path.name for path in stage1.glob("*.yaml")} == {
        "memory_add_wiring.yaml",
        "memory_tape32_wiring.yaml",
    }

    stage2 = ROOT / "configs" / "stage2"
    assert not list(stage2.glob("**/*.yaml"))
    locked = (
        ROOT
        / "experiments"
        / "stage2_training"
        / "main"
        / "LOCKED_PROTOCOL.md"
    ).read_text(encoding="utf-8")
    assert "Status: NOT LOCKED" in locked


def test_selected_starting_points_are_pinned():
    manifest_path = (
        ROOT / "experiments" / "stage1_starting_points" / "STARTING_POINTS.yaml"
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "selected"
    models = manifest["models"]

    assert models["memory_add"]["checkpoint"] == (
        "checkpoints/memory_add_frozen_wired_v1.pt"
    )
    assert models["memory_add"]["sha256"] == (
        "62885c820499b987ebf7949b81d6ef0de66e0327f1c9b75d7a865653a046248d"
    )
    assert models["memory_tape32"]["checkpoint"] == (
        "checkpoints/memory_tape32_frozen_wired_v1.pt"
    )
    assert models["memory_tape32"]["sha256"] == (
        "7799bb01bfbe112309007585bae7e9183b7d01d9df4f2454f640b1c126f9964f"
    )
    assert models["memory_add"]["backbone_identical_to_vanilla"] is True
    assert models["memory_tape32"]["backbone_identical_to_vanilla"] is True
    assert models["fbt"]["status"] == "not_selected_for_main_comparison"


def test_all_runnable_research_configs_parse():
    paths = _experiment_config_paths()
    assert paths
    for path in paths:
        load_experiment_config(path)


def test_stage2_k3_configs_are_development_not_canonical():
    base = (
        ROOT
        / "experiments"
        / "stage2_training"
        / "protocol_development"
        / "pass_depth"
        / "configs"
    )
    for variant in ("memory_add", "memory_tape32"):
        short = load_experiment_config(base / f"{variant}_k3_262k.yaml")
        continuation = load_experiment_config(base / f"{variant}_k3_continue_1m.yaml")
        assert short.normalized_pass_schedule() == [
            {"until_tokens": None, "probabilities": {3: 1.0}}
        ]
        assert short.pass_loss_weights == [0.1, 0.3, 0.6]
        assert short.max_unique_tokens == 262_144
        assert short.resume_from is None
        assert "selected-lr1e-7-long" in (short.init_from or "")

        assert continuation.max_unique_tokens == 1_048_576
        assert continuation.init_from is None
        assert continuation.resume_from == (
            f"runs/mac-{variant.replace('_', '-')}-phase-b-k3-short/latest.pt"
        )
