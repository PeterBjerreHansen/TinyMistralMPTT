from pathlib import Path

import torch

from tiny_mistral_mptt.data.packed_dataset import StatefulBlockSampler
from tiny_mistral_mptt.training.checkpoint import TrainState, load_checkpoint, save_checkpoint


def test_checkpoint_restores_model_optimizer_sampler_and_counters(tmp_path):
    torch.manual_seed(11)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    x = torch.randn(2, 4)
    model(x).sum().backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    sampler = StatefulBlockSampler(13, seed=7)
    sampler.next_indices(6)
    state = TrainState(optimizer_steps=1, micro_steps=2, unique_tokens_seen=64, token_equivalent_compute=64)
    path = save_checkpoint(
        tmp_path / "state.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=state,
        experiment_config={"variant": "vanilla"},
        data_manifest_sha256="manifest-hash",
    )
    expected_parameters = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    expected_next = sampler.next_indices(8)

    replacement = torch.nn.Linear(4, 3)
    replacement_optimizer = torch.optim.AdamW(replacement.parameters(), lr=9e-2, foreach=False)
    loaded_state, sampler_state = load_checkpoint(
        path,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_manifest_sha256="manifest-hash",
    )
    restored_sampler = StatefulBlockSampler(13, seed=0)
    restored_sampler.load_state_dict(sampler_state)
    assert loaded_state == state
    assert restored_sampler.next_indices(8) == expected_next
    for name, tensor in replacement.state_dict().items():
        torch.testing.assert_close(tensor, expected_parameters[name], atol=0, rtol=0)


def test_checkpoint_rejects_training_config_changes(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "state.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(),
        experiment_config={"variant": "vanilla", "batch_size": 1, "output_dir": "a", "resume_from": None},
        data_manifest_sha256="same",
    )
    replacement = torch.nn.Linear(2, 2)
    replacement_optimizer = torch.optim.AdamW(replacement.parameters(), lr=1e-3, foreach=False)
    import pytest
    with pytest.raises(ValueError, match="batch_size"):
        load_checkpoint(
            path,
            model=replacement,
            optimizer=replacement_optimizer,
            expected_manifest_sha256="same",
            expected_experiment_config={"variant": "vanilla", "batch_size": 2, "output_dir": "b", "resume_from": str(path)},
        )


def test_checkpoint_resume_allows_extending_token_budget(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "extend.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(unique_tokens_seen=4),
        experiment_config={
            "variant": "vanilla",
            "max_unique_tokens": 4,
            "output_dir": "a",
            "resume_from": None,
        },
        data_manifest_sha256="same",
    )
    replacement = torch.nn.Linear(2, 2)
    replacement_optimizer = torch.optim.AdamW(replacement.parameters(), lr=1e-3, foreach=False)
    state, _ = load_checkpoint(
        path,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_manifest_sha256="same",
        expected_experiment_config={
            "variant": "vanilla",
            "max_unique_tokens": 8,
            "output_dir": "extended",
            "resume_from": str(path),
        },
    )
    assert state.unique_tokens_seen == 4


def test_checkpoint_resume_accepts_new_default_experiment_fields(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "defaults.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(),
        experiment_config={"variant": "memory_tape32"},
        data_manifest_sha256="same",
    )
    replacement = torch.nn.Linear(2, 2)
    replacement_optimizer = torch.optim.AdamW(replacement.parameters(), lr=1e-3, foreach=False)
    load_checkpoint(
        path,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_manifest_sha256="same",
        expected_experiment_config={
            "variant": "memory_tape32",
            "prefix_mixin_probability": 0.0,
        },
    )


def test_checkpoint_resume_ignores_retired_fbt_calibration_metadata(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "legacy-calibration.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(),
        experiment_config={
            "variant": "fbt",
            "fbt_initialization": "calibrated",
            "fbt_calibration_split": "validation",
            "fbt_calibration_block": 0,
            "fbt_gate_logit_std_target": 1.0,
        },
        data_manifest_sha256="same",
    )
    replacement = torch.nn.Linear(2, 2)
    replacement_optimizer = torch.optim.AdamW(
        replacement.parameters(), lr=1e-3, foreach=False
    )
    load_checkpoint(
        path,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_manifest_sha256="same",
        expected_experiment_config={"variant": "fbt"},
    )


def test_checkpoint_resume_allows_new_output_and_evaluation_schedule(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "schedule.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(unique_tokens_seen=7),
        experiment_config={
            "variant": "memory_add",
            "phase": "B",
            "init_from": "source.pt",
            "eval_every_tokens": 65536,
            "eval_batches": 16,
            "eval_passes": 2,
            "checkpoint_every_tokens": 65536,
        },
        data_manifest_sha256="same",
    )
    replacement = torch.nn.Linear(2, 2)
    replacement_optimizer = torch.optim.AdamW(replacement.parameters(), lr=1e-3, foreach=False)
    state, _ = load_checkpoint(
        path,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_manifest_sha256="same",
        expected_experiment_config={
            "variant": "memory_add",
            "phase": "B",
            "init_from": None,
            "output_dir": "new-output",
            "resume_from": str(path),
            "max_unique_tokens": 1048576,
            "eval_every_tokens": 262144,
            "eval_batches": 32,
            "eval_passes": 8,
            "checkpoint_every_tokens": 262144,
        },
    )
    assert state.unique_tokens_seen == 7


def test_version1_checkpoint_allows_new_default_config_fields(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "state-v2.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(),
        experiment_config={"variant": "vanilla", "batch_size": 1, "output_dir": "a", "resume_from": None},
        data_manifest_sha256="same",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = 1
    payload.pop("pass_scheduler", None)
    v1 = tmp_path / "state-v1.pt"
    torch.save(payload, v1)

    replacement = torch.nn.Linear(2, 2)
    replacement_optimizer = torch.optim.AdamW(replacement.parameters(), lr=1e-3, foreach=False)
    state, _ = load_checkpoint(
        v1,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_manifest_sha256="same",
        expected_experiment_config={
            "variant": "vanilla",
            "batch_size": 1,
            "output_dir": "new",
            "resume_from": str(v1),
            "phase": "B",
            "pass_schedule": None,
        },
    )
    assert state == TrainState()
