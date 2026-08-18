from pathlib import Path

import yaml

from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.config import load_data_config

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RESULTS = (
    ROOT / "benchmarks" / "historical" / "stage2_cleanroom_v1" / "results"
)


def _active_config_paths() -> list[Path]:
    paths = list((ROOT / "benchmarks" / "controls" / "substrate").glob("*.yaml"))
    paths.extend((ROOT / "benchmarks" / "controls" / "smoke").glob("*.yaml"))
    paths.extend(
        path
        for path in (ROOT / "benchmarks" / "development").glob("**/*.yaml")
        if path.name not in {"PLAN.yaml", "data.yaml"}
    )
    return sorted(paths)


def test_locked_layout_is_explicit():
    assert not (ROOT / "configs").exists()
    assert not (ROOT / "experiments").exists()
    assert (ROOT / "benchmarks" / "controls").is_dir()
    assert (ROOT / "benchmarks" / "controls" / "smoke").is_dir()
    assert (ROOT / "benchmarks" / "controls" / "substrate").is_dir()
    assert (ROOT / "benchmarks" / "controls" / "smoke" / "results").is_dir()
    assert (ROOT / "benchmarks" / "controls" / "substrate" / "results").is_dir()
    assert (ROOT / "benchmarks" / "ad_hoc").is_dir()
    assert (ROOT / "benchmarks" / "development").is_dir()
    assert (ROOT / "benchmarks" / "core").is_dir()
    assert (ROOT / "benchmarks" / "historical").is_dir()

    assert (HISTORICAL_RESULTS / "k_sweep.md").exists()
    assert not (HISTORICAL_RESULTS.parent / "PROTOCOL.yaml").exists()
    assert not (HISTORICAL_RESULTS.parent / "configs").exists()
    assert not (ROOT / "runs" / "stage2_cleanroom_v1").exists()


def test_historical_results_are_retained_without_runnable_surface():
    expected = {
        "historical_comparison.md",
        "k_sweep.md",
        "learning_rate.md",
        "mixtures.md",
        "pass_depth_and_inference.md",
    }
    assert {path.name for path in HISTORICAL_RESULTS.glob("*.md")} == expected


def test_default_experiment_config_uses_active_2048_context():
    from tiny_mistral_mptt.config import ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.data_dir == "data/dolmino/local_2048"
    assert cfg.output_dir == "benchmarks/controls/smoke/results/vanilla"


def test_local_2048_data_recipe_parses():
    cfg = load_data_config(ROOT / "data" / "dolmino" / "local_2048" / "config.yaml")
    assert cfg.output_dir == "data/dolmino/local_2048"
    assert cfg.sequence_length == 2048
    assert cfg.train_tokens == 1_048_576
    assert cfg.validation_tokens == 131_072


def test_data_recipes_live_beside_their_artifacts():
    for name in ("local_2048", "gpu_2048"):
        path = ROOT / "data" / "dolmino" / name / "config.yaml"
        assert path.exists()
        cfg = load_data_config(path)
        assert cfg.output_dir == f"data/dolmino/{name}"
        assert cfg.sequence_length == 2048


def test_lm_evaluation_suites_live_with_the_data_assets():
    suite_dir = ROOT / "data" / "lm_evaluation"
    assert (suite_dir / "README.md").exists()
    assert (suite_dir / "results" / "README.md").exists()
    for name in ("quick.yaml", "full.yaml"):
        suite = yaml.safe_load((suite_dir / name).read_text(encoding="utf-8"))
        assert suite["tasks"]
        assert all("name" in task and "num_fewshot" in task for task in suite["tasks"])


def test_all_active_benchmark_configs_parse():
    paths = _active_config_paths()
    assert paths
    for path in paths:
        load_experiment_config(path)


def test_config_namespaces_are_current():
    roots = [
        ROOT / "benchmarks" / "controls",
        ROOT / "benchmarks" / "development",
        ROOT / "benchmarks" / "ad_hoc",
    ]
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in root.glob("**/*.yaml")
    )
    assert "runs/mac-" not in text
    assert "runs/cleanroom-v1" not in text
    assert "runs/" not in text
    assert "experiments/" not in text
    assert "configs/" not in text


def test_training_efficiency_defaults_to_2048_cases():
    suite = yaml.safe_load(
        (ROOT / "benchmarks" / "efficiency" / "suites" / "training.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert {case["sequence_length"] for case in suite["cases"]} == {2048}


def test_root_readme_keeps_stage2_protocol_pending():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "The Stage 2 protocol remains open" in readme
    assert "configs/" not in readme
    assert "backbone learning rate `3e-7`" not in readme


def test_stage2_docs_do_not_claim_an_old_lock():
    lr_report = (HISTORICAL_RESULTS / "learning_rate.md").read_text(encoding="utf-8")
    validation = (ROOT / "docs" / "VALIDATION.md").read_text(encoding="utf-8")
    assert "locked clean-room protocol remains unchanged" not in lr_report
    assert "historical clean-room result records" in validation
