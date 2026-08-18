#!/usr/bin/env python
"""Verify the selected-LR K-sweep provenance before launching it."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.config_invariants import (
    differing_invariant_fields,
    execution_invariant_view,
)
from tiny_mistral_mptt.data.manifest import file_sha256


ROOT = Path(__file__).resolve().parents[1]
LINEAGE = ROOT / "experiments" / "stage2_cleanroom_v1"

SCHEDULES = {
    "k2": ({2: 1.0}, [0.25, 0.75], None, 2),
    "k2_90_k3_10": (
        {2: 0.9, 3: 0.1},
        None,
        {2: [0.25, 0.75], 3: [0.05, 0.20, 0.75]},
        3,
    ),
    "k2_50_k3_50": (
        {2: 0.5, 3: 0.5},
        None,
        {2: [0.25, 0.75], 3: [0.05, 0.20, 0.75]},
        3,
    ),
    "k3": ({3: 1.0}, [0.05, 0.20, 0.75], None, 3),
}


def _git_status() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        text=True,
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="skip the clean-worktree requirement while developing the gate",
    )
    args = parser.parse_args()

    protocol_path = LINEAGE / "PROTOCOL.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    _assert(protocol["status"] == "k_schedule_pending", "protocol is not pending K selection")
    _assert(
        protocol["locked_protocol"]["backbone_learning_rate"] == 1.0e-6,
        "protocol backbone LR is not 1e-6",
    )
    _assert(
        protocol["locked_protocol"]["added_learning_rate"] == 1.0e-6,
        "protocol added-parameter LR is not 1e-6",
    )

    if not args.allow_dirty:
        _assert(not _git_status().strip(), "worktree is dirty")

    data_recipe = LINEAGE / "configs" / "data" / "artifact.yaml"
    _assert(data_recipe.exists(), f"missing data recipe: {data_recipe}")
    data_config = yaml.safe_load(data_recipe.read_text(encoding="utf-8"))
    manifest_path = ROOT / data_config["output_dir"] / "manifest.json"
    _assert(manifest_path.exists(), f"missing data manifest: {manifest_path}")
    _assert(
        file_sha256(manifest_path) == protocol["data"]["manifest_sha256"],
        "clean-room data manifest hash does not match PROTOCOL.yaml",
    )

    common_by_variant = {}
    expected_files = {f"{name}.yaml" for name in SCHEDULES}
    for variant in ("memory_add", "memory_tape32"):
        config_dir = LINEAGE / "configs" / "k_sweep" / variant
        paths = sorted(config_dir.glob("*.yaml"))
        _assert({path.name for path in paths} == expected_files, f"wrong {variant} K-sweep files")
        expected_parent = protocol["starting_points"][variant]["checkpoint"]
        expected_parent_path = ROOT / expected_parent
        _assert(expected_parent_path.exists(), f"missing E1 checkpoint: {expected_parent_path}")
        _assert(
            file_sha256(expected_parent_path) == protocol["starting_points"][variant]["sha256"],
            f"{variant} E1 checkpoint hash does not match PROTOCOL.yaml",
        )

        for path in paths:
            name = path.stem
            config = load_experiment_config(path)
            expected_schedule, expected_weights, expected_weights_by_k, expected_eval_passes = SCHEDULES[name]
            _assert(config.variant == variant, f"{path} has the wrong variant")
            _assert(config.phase == "B", f"{path} is not a Phase-B config")
            _assert(config.resume_from is None, f"{path} must not resume another K-sweep arm")
            fields = execution_invariant_view(config)
            reference = common_by_variant.setdefault(variant, fields)
            differences = differing_invariant_fields(reference, fields)
            _assert(not differences, f"{path} differs in execution invariants: {differences}")
            _assert(config.init_from == expected_parent, f"{path} has the wrong E1 parent")
            expected_output = ROOT / "runs" / "stage2_cleanroom_v1" / "k_sweep" / variant / f"{name}"
            actual_output = (ROOT / config.output_dir).resolve()
            _assert(actual_output == expected_output, f"{path} has the wrong output_dir")
            _assert(config.pass_schedule[0]["probabilities"] == expected_schedule, f"{path} has the wrong K schedule")
            _assert(config.pass_loss_weights == expected_weights, f"{path} has the wrong fixed-K weights")
            _assert(config.pass_loss_weights_by_k == expected_weights_by_k, f"{path} has the wrong K-specific weights")
            _assert(config.eval_passes == expected_eval_passes, f"{path} has the wrong validation pass count")

    print("PASS: selected-LR K-sweep configs, data, checkpoints, and provenance are consistent")


if __name__ == "__main__":
    main()
