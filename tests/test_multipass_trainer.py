from pathlib import Path

import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.config import ExperimentConfig
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.data.prepare import PreparationRequest, materialize_from_document_iterators
from tiny_mistral_mptt.data.recipes import DOLMINO_50B_SOURCES
from tiny_mistral_mptt.training.trainer import Trainer
from tiny_mistral_mptt.variants.fbt import FBTVariant


def fake_docs(offset: int):
    for doc in range(10_000):
        yield f"{offset}-{doc}-abcdefghijklmnopqrstuvwxyz"


def make_artifact(root: Path):
    materialize_from_document_iterators(
        PreparationRequest(
            output_dir=root,
            sequence_length=8,
            train_tokens=8 * 40,
            validation_tokens=8 * 8,
            seed=5,
            dataset_repo="fake",
            requested_revision="x",
            resolved_revision="x",
            tokenizer_file=Path("fake-tokenizer"),
            tokenizer_sha256="fake",
            vocab_size=97,
            bos_token_id=1,
        ),
        iterators={source.name: iter(fake_docs(i)) for i, source in enumerate(DOLMINO_50B_SOURCES)},
        tokenize=lambda text: [3 + (ord(ch) % 80) for ch in text],
    )


def make_fbt(seed=123):
    torch.manual_seed(17)
    return FBTVariant(
        MistralForCausalLM(micro_config(), attention_backend="reference"),
        initialization_seed=seed,
    )


def test_phase_a_fixed_two_pass_training_counts_compute_and_freezes_backbone(tmp_path):
    data_dir = tmp_path / "data"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    model = make_fbt()
    before = {name: tensor.detach().clone() for name, tensor in model.backbone.state_dict().items()}
    cfg = ExperimentConfig(
        variant="fbt",
        phase="A",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "phase-a"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=32,
        batch_size=1,
        grad_accum_steps=1,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        pass_loss_weights=[0.0, 1.0],
        added_learning_rate=1e-3,
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    state = Trainer(model=model, config=cfg, train_data=train, validation_data=val, device=torch.device("cpu")).train()
    assert state.unique_tokens_seen == 32
    assert state.token_equivalent_compute == 64
    for name, tensor in model.backbone.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


def test_phase_b_has_independent_pretrained_and_added_learning_rates(tmp_path):
    data_dir = tmp_path / "data-lr"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    cfg = ExperimentConfig(
        variant="fbt",
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "phase-b"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=16,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        pretrained_learning_rate=1e-6,
        added_learning_rate=1e-4,
        lr_schedule={"type": "constant"},
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    trainer = Trainer(model=make_fbt(), config=cfg, train_data=train, validation_data=val, device=torch.device("cpu"))
    groups = {group["group_name"]: group["base_lr"] for group in trainer.optimizer.param_groups}
    assert groups == {"pretrained": 1e-6, "added": 1e-4}


def test_init_from_loads_model_only_into_fresh_phase_b_run(tmp_path):
    data_dir = tmp_path / "data-init"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    phase_a_cfg = ExperimentConfig(
        variant="fbt",
        phase="A",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "a"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=16,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        pass_loss_weights=[0.0, 1.0],
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    phase_a_model = make_fbt(seed=7)
    Trainer(
        model=phase_a_model,
        config=phase_a_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()
    checkpoint = tmp_path / "a" / "latest.pt"
    expected = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]

    phase_b_cfg = ExperimentConfig(
        variant="fbt",
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "b"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=16,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        init_from=str(checkpoint),
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    fresh = make_fbt(seed=999)
    trainer = Trainer(
        model=fresh,
        config=phase_b_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    )
    assert trainer.state.unique_tokens_seen == 0
    assert trainer.state.optimizer_steps == 0
    for name, tensor in fresh.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)


def test_mixed_pass_schedule_resume_is_bit_exact(tmp_path):
    data_dir = tmp_path / "data-resume"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    common = dict(
        variant="fbt",
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        device="cpu",
        attention_backend="reference",
        seed=55,
        architecture_seed=77,
        batch_size=1,
        grad_accum_steps=1,
        max_unique_tokens=64,
        pass_schedule=[{"probabilities": {1: 0.4, 2: 0.4, 3: 0.2}}],
        pass_loss_weights=[0.05, 0.20, 0.75],
        pretrained_learning_rate=1e-4,
        added_learning_rate=1e-4,
        lr_schedule={"type": "constant"},
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )

    full = make_fbt(seed=77)
    full_cfg = ExperimentConfig(output_dir=str(tmp_path / "full"), **common)
    full_state = Trainer(
        model=full,
        config=full_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()

    interrupted = make_fbt(seed=77)
    interrupted_cfg = ExperimentConfig(output_dir=str(tmp_path / "interrupted"), **common)
    Trainer(
        model=interrupted,
        config=interrupted_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train(until_unique_tokens=32)
    checkpoint = tmp_path / "interrupted" / "latest.pt"

    resumed = make_fbt(seed=77)
    resumed_cfg = ExperimentConfig.from_dict(
        {
            **interrupted_cfg.to_dict(),
            "output_dir": str(tmp_path / "resumed"),
            "resume_from": str(checkpoint),
        }
    )
    resumed_state = Trainer(
        model=resumed,
        config=resumed_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()

    assert resumed_state == full_state
    for name, tensor in full.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[name], atol=0, rtol=0)


def test_memory_tape_phase_a_runs_through_shared_trainer(tmp_path):
    from tiny_mistral_mptt.variants.memory_tape32 import MemoryTape32Variant

    data_dir = tmp_path / "data-memory"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    torch.manual_seed(23)
    model = MemoryTape32Variant(
        MistralForCausalLM(micro_config(), attention_backend="reference"),
        memory_window=4,
        initialization_seed=31,
    )
    cfg = ExperimentConfig(
        variant="memory_tape32",
        phase="A",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "memory-a"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=16,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        pass_loss_weights=[0.0, 1.0],
        added_learning_rate=1e-3,
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    state = Trainer(
        model=model,
        config=cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()
    assert state.unique_tokens_seen == 16
    assert state.token_equivalent_compute == 32


def test_memory_add_phase_a_runs_through_shared_trainer(tmp_path):
    from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant

    data_dir = tmp_path / "data-memory-add"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    torch.manual_seed(29)
    model = MemoryAddVariant(
        MistralForCausalLM(micro_config(), attention_backend="reference")
    )
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.backbone.state_dict().items()
    }
    cfg = ExperimentConfig(
        variant="memory_add",
        phase="A",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "memory-add-a"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=16,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        pass_loss_weights=[0.0, 1.0],
        added_learning_rate=1e-3,
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    state = Trainer(
        model=model,
        config=cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()
    assert state.unique_tokens_seen == 16
    assert state.token_equivalent_compute == 32
    assert torch.count_nonzero(model.memory_projection.weight) > 0
    for name, tensor in model.backbone.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
