from pathlib import Path
import importlib.util


_RUN_STUDY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_study.py"
_SPEC = importlib.util.spec_from_file_location("run_study", _RUN_STUDY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUN_STUDY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN_STUDY)
_should_resume_auto = _RUN_STUDY._should_resume_auto


def _write_config(path: Path, *, output_dir: str, init_from: str | None) -> None:
    path.write_text(
        "\n".join(
            [
                f"output_dir: {output_dir}",
                f"init_from: {init_from if init_from is not None else 'null'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_init_from_config_starts_fresh_when_output_is_empty(tmp_path):
    config = tmp_path / "phase-b.yaml"
    _write_config(config, output_dir="results/phase-b", init_from="checkpoints/phase-a.pt")

    assert _should_resume_auto(config, root=tmp_path) is False


def test_init_from_config_resumes_when_output_has_trajectory(tmp_path):
    config = tmp_path / "phase-b.yaml"
    _write_config(config, output_dir="results/phase-b", init_from="checkpoints/phase-a.pt")
    output_dir = tmp_path / "results" / "phase-b"
    output_dir.mkdir(parents=True)
    (output_dir / "run.json").write_text("{}\n", encoding="utf-8")

    assert _should_resume_auto(config, root=tmp_path) is True


def test_config_without_init_from_keeps_auto_resume_behavior(tmp_path):
    config = tmp_path / "phase-a.yaml"
    _write_config(config, output_dir="results/phase-a", init_from=None)

    assert _should_resume_auto(config, root=tmp_path) is True
