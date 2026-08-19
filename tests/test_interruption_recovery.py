from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tiny_mistral_mptt.training.checkpoint import (
    TrainState,
    candidate_checkpoint_paths,
    discover_checkpoint_generations,
    load_checkpoint,
    load_latest_valid_checkpoint,
    save_checkpoint_generation,
)
from tiny_mistral_mptt.training.journal import append_jsonl, repair_metrics_to_checkpoint


def _objects(seed: int = 7):
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    return model, optimizer


def _advance(model, optimizer):
    x = torch.randn(2, 4)
    model(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _save(run: Path, model, optimizer, tokens: int, *, source=None):
    return save_checkpoint_generation(
        run,
        model=model,
        optimizer=optimizer,
        sampler_state={"position": tokens},
        train_state=TrainState(
            optimizer_steps=tokens // 8,
            micro_steps=tokens // 8,
            unique_tokens_seen=tokens,
            token_equivalent_compute=tokens,
        ),
        experiment_config={
            "variant": "vanilla",
            "batch_size": 1,
            "lr_schedule": {"type": "constant"},
            "checkpoint_every_tokens": 8,
            "checkpoint_every_seconds": 1.0,
            "checkpoint_keep_last": 2,
        },
        data_manifest_sha256="manifest",
        source_provenance=source,
        keep_last=2,
    )


def test_generation_retention_keeps_exactly_two(tmp_path):
    model, optimizer = _objects()
    _save(tmp_path, model, optimizer, 8)
    _advance(model, optimizer)
    _save(tmp_path, model, optimizer, 16)
    _advance(model, optimizer)
    _save(tmp_path, model, optimizer, 24)

    generations = discover_checkpoint_generations(tmp_path)
    assert [path.name for path in generations] == [
        "checkpoint_000000000024.pt",
        "checkpoint_000000000016.pt",
    ]
    pointer = json.loads((tmp_path / "checkpoints" / "latest.json").read_text())
    assert pointer["current"] == "checkpoint_000000000024.pt"
    assert pointer["previous"] == "checkpoint_000000000016.pt"
    assert (tmp_path / "latest.pt").resolve() == generations[0].resolve()


def test_corrupt_newest_falls_back_to_previous(tmp_path):
    model, optimizer = _objects()
    _save(tmp_path, model, optimizer, 8)
    _advance(model, optimizer)
    previous = _save(tmp_path, model, optimizer, 16)
    expected = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    _advance(model, optimizer)
    newest = _save(tmp_path, model, optimizer, 24)
    newest.write_bytes(b"truncated checkpoint")

    replacement, replacement_optimizer = _objects(seed=999)
    path, state, sampler, fallback = load_latest_valid_checkpoint(
        tmp_path,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_manifest_sha256="manifest",
        expected_experiment_config={
            "variant": "vanilla",
            "batch_size": 1,
            "lr_schedule": {"type": "constant"},
            "checkpoint_every_tokens": 100,
            "checkpoint_every_seconds": 600,
            "checkpoint_keep_last": 2,
        },
    )
    assert path == previous
    assert fallback is True
    assert state.unique_tokens_seen == 16
    assert sampler == {"position": 16}
    for name, tensor in replacement.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)


def test_uncommitted_tmp_is_ignored(tmp_path):
    model, optimizer = _objects()
    current = _save(tmp_path, model, optimizer, 8)
    stray = tmp_path / "checkpoints" / "checkpoint_000000000016.pt.tmp"
    stray.write_bytes(b"half-written")
    assert discover_checkpoint_generations(tmp_path) == [current]
    assert candidate_checkpoint_paths(tmp_path)[0] == current


def test_metrics_repair_rolls_back_to_checkpoint(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_jsonl(path, {"event": "train", "optimizer_steps": 1, "unique_tokens_seen": 8})
    append_jsonl(path, {"event": "validation", "optimizer_steps": 1, "unique_tokens_seen": 8})
    append_jsonl(path, {"event": "train", "optimizer_steps": 2, "unique_tokens_seen": 16})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event":"train"')

    result = repair_metrics_to_checkpoint(
        path,
        TrainState(optimizer_steps=1, micro_steps=1, unique_tokens_seen=8, token_equivalent_compute=8),
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert result == {"kept": 2, "dropped": 1, "malformed": 1}
    assert [record["event"] for record in records] == ["train", "validation"]
    assert max(record["unique_tokens_seen"] for record in records) == 8


def test_format_v3_rejects_source_identity_change(tmp_path):
    model, optimizer = _objects()
    checkpoint = _save(
        tmp_path,
        model,
        optimizer,
        8,
        source={"git_commit": "a" * 40, "uv_lock_sha256": "1" * 64, "source_code_sha256": "2" * 64},
    )
    replacement, replacement_optimizer = _objects(seed=999)
    with pytest.raises(ValueError, match="source commit, uv.lock, or execution-code hash"):
        load_checkpoint(
            checkpoint,
            model=replacement,
            optimizer=replacement_optimizer,
            expected_manifest_sha256="manifest",
            expected_experiment_config={
                "variant": "vanilla",
                "batch_size": 1,
                "lr_schedule": {"type": "constant"},
            },
            expected_source_provenance={
                "git_commit": "b" * 40,
                "uv_lock_sha256": "1" * 64,
                "source_code_sha256": "3" * 64,
            },
        )


def test_format_v3_accepts_same_code_hash_with_different_git_metadata(tmp_path):
    model, optimizer = _objects()
    checkpoint = _save(
        tmp_path, model, optimizer, 8,
        source={"git_commit": "a" * 40, "uv_lock_sha256": "1" * 64, "source_code_sha256": "2" * 64},
    )
    replacement, replacement_optimizer = _objects(seed=999)
    state, _ = load_checkpoint(
        checkpoint,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_manifest_sha256="manifest",
        expected_experiment_config={"variant": "vanilla", "batch_size": 1, "lr_schedule": {"type": "constant"}},
        expected_source_provenance={"git_commit": None, "uv_lock_sha256": "1" * 64, "source_code_sha256": "2" * 64},
    )
    assert state.unique_tokens_seen == 8


def test_pending_validation_metadata_survives_checkpoint(tmp_path):
    from tiny_mistral_mptt.training.checkpoint import inspect_checkpoint

    model, optimizer = _objects()
    path = save_checkpoint_generation(
        tmp_path,
        model=model,
        optimizer=optimizer,
        sampler_state={"position": 8},
        train_state=TrainState(optimizer_steps=1, micro_steps=1, unique_tokens_seen=8, token_equivalent_compute=8),
        experiment_config={"variant": "vanilla", "lr_schedule": {"type": "constant"}},
        data_manifest_sha256="manifest",
        checkpoint_metadata={"pending_validation": True},
        keep_last=2,
    )
    assert inspect_checkpoint(path)["checkpoint_metadata"] == {"pending_validation": True}
