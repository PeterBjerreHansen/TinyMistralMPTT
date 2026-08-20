import torch

from tiny_mistral import MistralConfig, MistralForCausalLM
from tiny_mistral_mptt.feedback import TapeState
from tiny_mistral_mptt.variants import TapeReader, TapeVariant


def make_backbone() -> MistralForCausalLM:
    torch.manual_seed(1)
    config = MistralConfig(
        vocab_size=31,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=6,
        sliding_window=None,
        use_cache=True,
    )
    return MistralForCausalLM(config)


def test_raw_and_projected_bank_match() -> None:
    backbone = make_backbone()
    reader = TapeReader(backbone, window=4, initialization_seed=7)
    query = torch.randn(2, 1, 24)
    memories = torch.randn(2, 4, 24)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    raw = reader.forward_bank(query, memories, memory_mask=mask)
    key, value = reader.project_memory(memories)
    projected = reader.forward_projected_bank(query, key, value, memory_mask=mask)

    torch.testing.assert_close(raw, projected, rtol=0, atol=0)


def test_seeded_state_projection_matches_readers() -> None:
    backbone = make_backbone()
    model = TapeVariant(
        backbone,
        memory_window=4,
        memory_write_mode="dense",
        memory_write_stride=1,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    hidden = torch.randn(1, 3, 24)
    state = model._feedback_memory_from_hidden(hidden, input_ids=input_ids)

    assert isinstance(state, TapeState)
    assert state.projected_keys is not None
    assert state.projected_values is not None
    for index, reader in enumerate(model.memory_readers):
        key, value = reader.project_memory(state.memories)
        torch.testing.assert_close(state.projected_keys[index], key)
        torch.testing.assert_close(state.projected_values[index], value)


def test_periodic_nonwrite_preserves_projected_cache_exactly() -> None:
    model = TapeVariant(
        make_backbone(),
        memory_window=4,
        memory_write_mode="periodic",
        memory_write_stride=4,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    hidden = torch.randn(1, 4, 24)
    state = model._feedback_memory_from_hidden(hidden, input_ids=input_ids)
    updated = model._append_feedback_memory(
        state,
        torch.randn(1, 1, 24),
        token=torch.tensor([[5]]),
        position=4,
    )

    assert torch.equal(updated.memories, state.memories)
    assert torch.equal(updated.valid, state.valid)
    assert updated.projected_keys is not None and state.projected_keys is not None
    assert updated.projected_values is not None and state.projected_values is not None
    for new, old in zip(updated.projected_keys, state.projected_keys, strict=True):
        assert torch.equal(new, old)
    for new, old in zip(updated.projected_values, state.projected_values, strict=True):
        assert torch.equal(new, old)


def test_append_and_eviction_keep_raw_kv_aligned() -> None:
    model = TapeVariant(
        make_backbone(),
        memory_window=2,
        memory_write_mode="dense",
        memory_write_stride=1,
    )
    input_ids = torch.tensor([[1, 2]])
    hidden = torch.randn(1, 2, 24)
    state = model._feedback_memory_from_hidden(hidden, input_ids=input_ids)
    new_hidden = torch.randn(1, 1, 24)
    updated = model._append_feedback_memory(
        state,
        new_hidden,
        token=torch.tensor([[3]]),
        position=2,
    )
    expected_new = model.writer(new_hidden).detach()

    torch.testing.assert_close(updated.memories[:, 0], state.memories[:, 1])
    torch.testing.assert_close(updated.memories[:, 1:], expected_new)
    assert updated.projected_keys is not None and updated.projected_values is not None
    for index, reader in enumerate(model.memory_readers):
        key, value = reader.project_memory(updated.memories)
        torch.testing.assert_close(updated.projected_keys[index], key)
        torch.testing.assert_close(updated.projected_values[index], value)


def test_cached_read_does_not_reproject_old_memory(monkeypatch) -> None:
    model = TapeVariant(
        make_backbone(),
        memory_window=4,
        memory_write_mode="dense",
        memory_write_stride=1,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    state = model._feedback_memory_from_hidden(
        torch.randn(1, 3, 24), input_ids=input_ids
    )
    assert state.projected_keys is not None and state.projected_values is not None

    counts = {"k": 0, "v": 0}
    reader = model.memory_readers[0]
    original_key = reader.k_proj.forward
    original_value = reader.v_proj.forward
    monkeypatch.setattr(
        reader.k_proj,
        "forward",
        lambda value: (counts.__setitem__("k", counts["k"] + 1) or original_key(value)),
    )
    monkeypatch.setattr(
        reader.v_proj,
        "forward",
        lambda value: (counts.__setitem__("v", counts["v"] + 1) or original_value(value)),
    )

    query = torch.randn(1, 1, 24)
    for _ in range(3):
        reader.forward_projected_bank(
            query,
            state.projected_keys[0],
            state.projected_values[0],
            memory_mask=state.valid,
        )
    assert counts == {"k": 0, "v": 0}
