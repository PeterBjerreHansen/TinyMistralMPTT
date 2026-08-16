from pathlib import Path

import torch
from conftest import micro_config

from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.data.prepare import (
    PreparationRequest,
    materialize_from_document_iterators,
)
from tiny_mistral_mptt.data.recipes import DOLMINO_50B_SOURCES
from tiny_mistral_mptt.evaluation.pass_depth import evaluate_pass_depth
from tiny_mistral_mptt.variants.fbt import FBTVariant


def fake_docs(offset: int):
    for doc in range(1000):
        yield f"{offset}-{doc}-abcdefghijklmnopqrstuvwxyz"


def make_artifact(root: Path):
    materialize_from_document_iterators(
        PreparationRequest(
            output_dir=root,
            sequence_length=8,
            train_tokens=8 * 16,
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
        iterators={
            source.name: iter(fake_docs(i))
            for i, source in enumerate(DOLMINO_50B_SOURCES)
        },
        tokenize=lambda text: [3 + (ord(ch) % 80) for ch in text],
    )


def test_pass_depth_evaluator_reports_each_pass_and_hidden_deltas(tmp_path):
    data_dir = tmp_path / "data"
    make_artifact(data_dir)
    dataset = PackedTokenDataset(data_dir, "validation")
    torch.manual_seed(9)
    model = FBTVariant(
        MistralForCausalLM(micro_config(), attention_backend="reference"),
        initialization_seed=10,
    )
    result = evaluate_pass_depth(model, dataset, device="cpu", passes=3, max_blocks=2)
    assert result.blocks == 2
    assert result.predicted_tokens == 2 * 7
    assert len(result.nll_by_pass) == 3
    assert len(result.perplexity_by_pass) == 3
    assert len(result.hidden_delta_rms) == 2
    assert all(value >= 0 for value in result.hidden_delta_rms)
