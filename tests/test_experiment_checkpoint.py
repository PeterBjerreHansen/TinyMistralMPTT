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
