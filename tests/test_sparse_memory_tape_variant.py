import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.feedback import SparseTapeState
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.memory_tape32 import MemoryTape32Variant
from tiny_mistral_mptt.variants.sparse_memory_tape import SparseMemoryTapeVariant


def make_backbone(seed=101, *, layers=2, sliding_window=8):
    torch.manual_seed(seed)
    return MistralForCausalLM(
        micro_config(num_hidden_layers=layers, sliding_window=sliding_window),
        attention_backend="reference",
    )


def make_sparse(*, stride=2, window=4, mode="periodic", token_id=None):
    return SparseMemoryTapeVariant(
        make_backbone(),
        memory_window=window,
        memory_write_mode=mode,
        memory_write_stride=stride,
        memory_token_id=token_id,
        initialization_seed=321,
    )


def test_periodic_write_mask_uses_completed_stride_positions():
    model = make_sparse(stride=4)
    ids = torch.arange(10)[None, :]
    mask = model.write_mask(ids)
    assert mask.tolist() == [[False, False, False, True, False, False, False, True, False, False]]


def test_token_write_mask_is_per_example():
    model = make_sparse(mode="token", token_id=7)
    ids = torch.tensor([[7, 1, 2, 7], [1, 7, 2, 3]])
    assert model.write_mask(ids).tolist() == [
        [True, False, False, True],
        [False, True, False, False],
    ]


def test_c1_identity_writer_is_exact_dense_memory_tape_bridge():
    backbone_dense = make_backbone(seed=5)
    backbone_sparse = make_backbone(seed=99)
    backbone_sparse.load_state_dict(backbone_dense.state_dict())
    dense = MemoryTape32Variant(
        backbone_dense, memory_window=4, initialization_seed=777
    ).eval()
    sparse = SparseMemoryTapeVariant(
        backbone_sparse,
        memory_window=4,
        memory_write_mode="periodic",
        memory_write_stride=1,
        initialization_seed=777,
    ).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        for passes in (2, 3):
            expected = dense.compute_passes(ids, passes=passes).final.hidden_states
            actual = sparse.compute_passes(ids, passes=passes).final.hidden_states
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_periodic_write_is_invisible_at_own_position():
    model = make_sparse(stride=4, window=4).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    embeddings = model.backbone.model.embed_tokens(ids)
    previous = torch.randn_like(embeddings)
    perturbed = previous.clone()
    perturbed[:, 3, :] += 3.0
    with torch.no_grad():
        base = model._run_feedback_hidden(ids, embeddings, previous)
        changed = model._run_feedback_hidden(ids, embeddings, perturbed)
    torch.testing.assert_close(changed[:, :4, :], base[:, :4, :], atol=0, rtol=0)
    assert not torch.allclose(changed[:, 4:, :], base[:, 4:, :])


def test_token_mode_with_no_write_events_is_exact_vanilla_feedback_pass():
    model = make_sparse(mode="token", token_id=31).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    embeddings = model.backbone.model.embed_tokens(ids)
    previous = torch.randn_like(embeddings)
    assert not model.write_mask(ids).any()
    with torch.no_grad():
        expected = model.backbone.model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
        actual = model._run_feedback_hidden(ids, embeddings, previous)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_phase_a_trains_writer_and_readers_but_not_backbone():
    model = make_sparse(stride=2)
    configure_phase(model, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = model.compute_loss(ids, phase="A", passes=2, loss_weights=[0.0, 1.0])
    output.loss.backward()
    assert model.writer.proj.weight.grad is not None
    assert torch.isfinite(model.writer.proj.weight.grad).all()
    assert model.writer.proj.weight.grad.abs().sum() > 0
    added = {id(parameter) for parameter in model.added_parameters()}
    for parameter in model.parameters():
        if id(parameter) not in added:
            assert parameter.grad is None


def test_seeded_sparse_feedback_state_keeps_last_w_committed_records():
    model = make_sparse(stride=2, window=3).eval()
    hidden = torch.arange(1 * 8 * model.config.hidden_size, dtype=torch.float32).reshape(
        1, 8, model.config.hidden_size
    )
    ids = torch.arange(8)[None, :]
    state = model._feedback_memory_from_hidden(hidden, input_ids=ids)
    assert isinstance(state, SparseTapeState)
    assert state.valid.tolist() == [[True, True, True]]
    # Writes occurred at positions 1,3,5,7; the retained tail is 3,5,7.
    torch.testing.assert_close(state.memories[0], hidden[0, [3, 5, 7], :])


def test_cached_periodic_tape_writes_only_after_trigger_position():
    model = make_sparse(stride=4, window=3).eval()
    dim = model.config.hidden_size
    state = SparseTapeState(
        memories=torch.zeros(1, 3, dim),
        valid=torch.zeros(1, 3, dtype=torch.bool),
    )
    hidden = torch.ones(1, 1, dim)
    token = torch.tensor([[2]])
    for position in range(3):
        state = model._append_feedback_memory(state, hidden, token=token, position=position)
        assert not state.valid.any()
    state = model._append_feedback_memory(state, hidden, token=token, position=3)
    assert state.valid.tolist() == [[True, False, False]]


def test_cached_token_writes_can_differ_across_batch_examples():
    model = make_sparse(mode="token", token_id=7, window=3).eval()
    dim = model.config.hidden_size
    state = SparseTapeState(
        memories=torch.zeros(2, 3, dim),
        valid=torch.zeros(2, 3, dtype=torch.bool),
    )
    hidden = torch.stack((torch.ones(1, dim), 2 * torch.ones(1, dim)), dim=0)
    token = torch.tensor([[7], [3]])
    state = model._append_feedback_memory(state, hidden, token=token, position=0)
    assert state.valid.tolist() == [[True, False, False], [False, False, False]]
