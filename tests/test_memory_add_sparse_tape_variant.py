import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.feedback import HybridFeedbackState
from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant
from tiny_mistral_mptt.variants.memory_add_sparse_tape import MemoryAddSparseTapeVariant
from tiny_mistral_mptt.variants.sparse_memory_tape import SparseMemoryTapeVariant


def backbone(seed):
    torch.manual_seed(seed)
    return MistralForCausalLM(
        micro_config(num_hidden_layers=2, sliding_window=8),
        attention_backend="reference",
    )


def hybrid_from_backbone(bb):
    return MemoryAddSparseTapeVariant(
        bb,
        memory_window=4,
        memory_write_stride=2,
        initialization_seed=555,
    )


def test_zero_fast_path_reduces_hybrid_exactly_to_sparse_tape():
    b1 = backbone(1)
    b2 = backbone(2)
    b2.load_state_dict(b1.state_dict())
    sparse = SparseMemoryTapeVariant(
        b1, memory_window=4, memory_write_stride=2, initialization_seed=555
    ).eval()
    hybrid = hybrid_from_backbone(b2).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        a = sparse.compute_passes(ids, passes=3).final.hidden_states
        b = hybrid.compute_passes(ids, passes=3).final.hidden_states
    torch.testing.assert_close(b, a, atol=0, rtol=0)


def test_zero_tape_path_reduces_hybrid_exactly_to_memory_add():
    b1 = backbone(4)
    b2 = backbone(5)
    b2.load_state_dict(b1.state_dict())
    add = MemoryAddVariant(b1).eval()
    hybrid = hybrid_from_backbone(b2).eval()
    dim = hybrid.config.hidden_size
    with torch.no_grad():
        weight = 0.04 * torch.eye(dim)
        add.memory_projection.weight.copy_(weight)
        hybrid.memory_projection.weight.copy_(weight)
        add.memory_norm.weight.copy_(hybrid.memory_norm.weight)
        for reader in hybrid.memory_readers:
            reader.o_proj.weight.zero_()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        a = add.compute_passes(ids, passes=3).final.hidden_states
        b = hybrid.compute_passes(ids, passes=3).final.hidden_states
    torch.testing.assert_close(b, a, atol=0, rtol=0)


def test_hybrid_feedback_state_updates_fast_every_token_and_tape_only_on_write():
    model = hybrid_from_backbone(backbone(8)).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        first = model._run_first_hidden(ids[:, :3])
        state = model._feedback_memory_from_hidden(first, input_ids=ids[:, :3])
        assert isinstance(state, HybridFeedbackState)
        old_valid = state.tape.valid.clone()
        new_hidden = torch.randn_like(state.fast_hidden)
        # Position 3 closes a C=2 interval and therefore writes.
        updated = model._append_feedback_memory(
            state, new_hidden, token=ids[:, 3:4], position=3
        )
        torch.testing.assert_close(updated.fast_hidden, new_hidden, atol=0, rtol=0)
        assert updated.tape.valid.sum() == old_valid.sum() + 1
        newer_hidden = torch.randn_like(state.fast_hidden)
        unchanged_tape = model._append_feedback_memory(
            updated, newer_hidden, token=ids[:, 4:5], position=4
        )
        torch.testing.assert_close(unchanged_tape.fast_hidden, newer_hidden, atol=0, rtol=0)
        torch.testing.assert_close(
            unchanged_tape.tape.memories, updated.tape.memories, atol=0, rtol=0
        )
        assert torch.equal(unchanged_tape.tape.valid, updated.tape.valid)


def test_hybrid_phase_a_trains_both_channels_with_backbone_frozen():
    from tiny_mistral_mptt.training.phases import configure_phase

    model = hybrid_from_backbone(backbone(13))
    configure_phase(model, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = model.compute_loss(ids, phase="A", passes=2, loss_weights=[0.0, 1.0])
    output.loss.backward()
    assert model.memory_projection.weight.grad is not None
    assert torch.isfinite(model.memory_projection.weight.grad).all()
    assert model.writer.proj.weight.grad is not None
    assert torch.isfinite(model.writer.proj.weight.grad).all()
    assert model.memory_readers[0].q_proj.weight.grad is not None
    added = {id(parameter) for parameter in model.added_parameters()}
    for parameter in model.parameters():
        if id(parameter) not in added:
            assert parameter.grad is None
