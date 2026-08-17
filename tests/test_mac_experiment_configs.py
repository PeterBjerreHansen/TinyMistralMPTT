from pathlib import Path

from tiny_mistral_mptt.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
MAC_CONFIGS = ROOT / "configs" / "mac"


def test_canonical_mac_configs_preserve_active_experiment_surface():
    expected = {
        "fbt_phase_a.yaml",
        "fbt_phase_b.yaml",
        "memory_add_phase_a.yaml",
        "memory_add_phase_b.yaml",
        "memory_add_phase_b_k3.yaml",
        "memory_add_phase_b_k3_short.yaml",
        "memory_tape32_phase_a.yaml",
        "memory_tape32_phase_b.yaml",
        "memory_tape32_phase_b_k3.yaml",
        "memory_tape32_phase_b_k3_short.yaml",
        "vanilla.yaml",
    }
    assert {path.name for path in MAC_CONFIGS.glob("*.yaml")} == expected

    configs = {
        name: load_experiment_config(MAC_CONFIGS / name)
        for name in expected
    }

    assert configs["fbt_phase_b.yaml"].init_from == "runs/mac-fbt-phase-a/latest.pt"
    assert configs["memory_add_phase_b.yaml"].init_from == (
        "checkpoints/memory_add_frozen_wired_v1.pt"
    )
    assert configs["memory_tape32_phase_b.yaml"].init_from == (
        "checkpoints/memory_tape32_frozen_wired_v1.pt"
    )

    for variant in ("memory_add", "memory_tape32"):
        short = configs[f"{variant}_phase_b_k3_short.yaml"]
        continuation = configs[f"{variant}_phase_b_k3.yaml"]
        assert short.pass_schedule == [{"probabilities": {3: 1.0}}]
        assert short.max_unique_tokens == 262_144
        assert short.resume_from is None
        assert short.init_from == (
            f"runs/mac-{variant.replace('_', '-')}-phase-b-selected-lr1e-7-long/latest.pt"
        )
        assert continuation.max_unique_tokens == 1_048_576
        assert continuation.resume_from == (
            f"runs/mac-{variant.replace('_', '-')}-phase-b-k3-short/latest.pt"
        )
        assert continuation.init_from is None


def test_active_k3_short_configs_match_archived_provenance():
    for variant in ("memory_add", "memory_tape32"):
        active = MAC_CONFIGS / f"{variant}_phase_b_k3_short.yaml"
        archived = (
            ROOT
            / "experiments"
            / "memory_phase_b"
            / "configs"
            / variant
            / f"{variant}_phase_b_k3_short.yaml"
        )
        assert active.read_bytes() == archived.read_bytes()
