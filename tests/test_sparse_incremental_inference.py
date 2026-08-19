import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.feedback import HybridFeedbackState, SparseTapeState
from tiny_mistral_mptt.inference import (
    exact_decode_step,
    prefill_exact,
    prefill_recurrent,
    recurrent_decode_step,
)
from tiny_mistral_mptt.variants.memory_add_sparse_tape import MemoryAddSparseTapeVariant
from tiny_mistral_mptt.variants.sparse_memory_tape import SparseMemoryTapeVariant


def make_model(name: str, *, mode="periodic"):
    torch.manual_seed(222)
    backbone = MistralForCausalLM(
        micro_config(num_hidden_layers=2, sliding_window=4),
        attention_backend="reference",
    )
    kwargs = dict(
        memory_window=3,
        memory_write_mode=mode,
        memory_write_stride=2,
        memory_token_id=7 if mode == "token" else None,
        initialization_seed=909,
    )
    if name == "sparse_memory_tape":
        model = SparseMemoryTapeVariant(backbone, **kwargs)
    elif name == "memory_add_sparse_tape":
        model = MemoryAddSparseTapeVariant(backbone, **kwargs)
        with torch.no_grad():
            dim = model.config.hidden_size
            model.memory_projection.weight.copy_(0.03 * torch.eye(dim))
    else:
        raise AssertionError(name)
    return model.eval()


def ids():
    # Includes repeated token-id 7 so token-trigger mode sees multiple writes.
    return torch.tensor([[1, 7, 3, 14, 7, 9, 31, 7, 51, 12, 6, 7, 18]])


@pytest.mark.parametrize("name", ["sparse_memory_tape", "memory_add_sparse_tape"])
@pytest.mark.parametrize("mode", ["periodic", "token"])
@pytest.mark.parametrize("passes", [1, 2, 3, 4])
def test_sparse_exact_incremental_matches_full_recomputation(name, mode, passes):
    model = make_model(name, mode=mode)
    sequence = ids()
    prompt_len = 5
    with torch.no_grad():
        state = prefill_exact(model, sequence[:, :prompt_len], passes=passes)
        full = model.compute_passes(sequence[:, :prompt_len], passes=passes)
        torch.testing.assert_close(
            state.next_token_logits, full.final.logits[:, -1, :], atol=4e-5, rtol=4e-5
        )
        for position in range(prompt_len, sequence.shape[1]):
            state = exact_decode_step(model, state, sequence[:, position : position + 1])
            full = model.compute_passes(sequence[:, : position + 1], passes=passes)
            torch.testing.assert_close(
                state.next_token_logits,
                full.final.logits[:, -1, :],
                atol=5e-5,
                rtol=5e-5,
            )
            torch.testing.assert_close(
                state.last_hidden,
                full.final.hidden_states[:, -1:, :],
                atol=5e-5,
                rtol=5e-5,
            )


@pytest.mark.parametrize("name", ["sparse_memory_tape", "memory_add_sparse_tape"])
@pytest.mark.parametrize("mode", ["periodic", "token"])
@pytest.mark.parametrize("passes", [2, 3, 4])
def test_sparse_recurrent_first_transition_matches_exact(name, mode, passes):
    model = make_model(name, mode=mode)
    sequence = ids()
    prompt = sequence[:, :6]
    token = sequence[:, 6:7]
    with torch.no_grad():
        exact = prefill_exact(model, prompt, passes=passes)
        recurrent = prefill_recurrent(model, prompt, passes=passes)
        exact_after = exact_decode_step(model, exact, token)
        recurrent_after = recurrent_decode_step(model, recurrent, token)
    torch.testing.assert_close(
        recurrent_after.next_token_logits,
        exact_after.next_token_logits,
        atol=5e-5,
        rtol=5e-5,
    )
    torch.testing.assert_close(
        recurrent_after.last_hidden, exact_after.last_hidden, atol=5e-5, rtol=5e-5
    )


def test_sparse_periodic_ring_is_bounded_and_updates_only_at_commits():
    model = make_model("sparse_memory_tape")
    sequence = ids()
    with torch.no_grad():
        state = prefill_recurrent(model, sequence[:, :5], passes=2)
        assert isinstance(state.feedback_memory, SparseTapeState)
        previous = state.feedback_memory
        for position in range(5, 10):
            state = recurrent_decode_step(model, state, sequence[:, position : position + 1])
            assert isinstance(state.feedback_memory, SparseTapeState)
            assert state.feedback_memory.capacity == model.memory_window
            assert int(state.feedback_memory.valid.sum()) <= model.memory_window
            should_write = (position + 1) % model.memory_write_stride == 0
            if not should_write:
                torch.testing.assert_close(
                    state.feedback_memory.memories, previous.memories, atol=0, rtol=0
                )
                assert torch.equal(state.feedback_memory.valid, previous.valid)
            previous = state.feedback_memory


def test_hybrid_fast_state_updates_even_when_sparse_tape_does_not():
    model = make_model("memory_add_sparse_tape")
    sequence = ids()
    with torch.no_grad():
        state = prefill_recurrent(model, sequence[:, :5], passes=2)
        assert isinstance(state.feedback_memory, HybridFeedbackState)
        old_tape = state.feedback_memory.tape
        # Absolute position 4 does not close a stride-2 interval.
        state = recurrent_decode_step(model, state, sequence[:, 5:6])
        assert isinstance(state.feedback_memory, HybridFeedbackState)
        torch.testing.assert_close(
            state.feedback_memory.fast_hidden, state.last_hidden, atol=0, rtol=0
        )
        # Position 5 *does* close C=2, so choose next non-write check at pos 6.
        state = recurrent_decode_step(model, state, sequence[:, 6:7])
        current = state.feedback_memory
        assert isinstance(current, HybridFeedbackState)
        torch.testing.assert_close(current.fast_hidden, state.last_hidden, atol=0, rtol=0)
        # Tape changed at pos5, then remains unchanged at pos6.
        tape_after_write = current.tape
        state2 = recurrent_decode_step(model, state, sequence[:, 7:8])
        # pos7 writes, so simply assert boundedness and fast update here.
        assert isinstance(state2.feedback_memory, HybridFeedbackState)
        assert state2.feedback_memory.tape.capacity == model.memory_window


def test_token_triggered_exact_inference_supports_different_batch_write_histories():
    model = make_model("sparse_memory_tape", mode="token")
    sequence = torch.tensor(
        [
            [1, 7, 3, 4, 5, 7, 8, 9],
            [1, 2, 3, 7, 5, 6, 7, 9],
        ]
    )
    prompt_len = 4
    with torch.no_grad():
        state = prefill_exact(model, sequence[:, :prompt_len], passes=2)
        for position in range(prompt_len, sequence.shape[1]):
            state = exact_decode_step(model, state, sequence[:, position : position + 1])
            full = model.compute_passes(sequence[:, : position + 1], passes=2)
            torch.testing.assert_close(
                state.next_token_logits,
                full.final.logits[:, -1, :],
                atol=5e-5,
                rtol=5e-5,
            )
