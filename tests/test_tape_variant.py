import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.data.packed_dataset import (
    insert_memory_tokens,
    memory_token_physical_length,
)
from tiny_mistral_mptt.feedback import HybridFeedbackState, TapeState
from tiny_mistral_mptt.training.loss import causal_lm_loss_from_labels
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.tape import TapeVariant
from tiny_mistral_mptt.variants.tape_add_hybrid import TapeAddHybridVariant


def backbone(seed=123, *, backend="reference", sliding_window=8):
    torch.manual_seed(seed)
    return MistralForCausalLM(
        micro_config(num_hidden_layers=2, sliding_window=sliding_window),
        attention_backend=backend,
        compile_flex=False,
    )


def tape_model(
    *,
    mode="periodic",
    stride=2,
    window=4,
    visibility="visible",
    hybrid=False,
    seed=123,
    backend="reference",
):
    cls = TapeAddHybridVariant if hybrid else TapeVariant
    model = cls(
        backbone(seed, backend=backend),
        memory_window=window,
        memory_write_mode=mode,
        memory_write_stride=stride,
        memory_token_visibility=visibility,
        initialization_seed=991,
    )
    if hybrid:
        with torch.no_grad():
            model.memory_projection.weight.copy_(0.03 * torch.eye(model.config.hidden_size))
    return model


def test_dense_and_periodic_c1_are_the_same_tape_architecture():
    dense = tape_model(mode="dense", stride=1, seed=5).eval()
    periodic = tape_model(mode="periodic", stride=1, seed=99).eval()
    periodic.backbone.load_state_dict(dense.backbone.state_dict())
    periodic.writer.load_state_dict(dense.writer.state_dict())
    periodic.memory_readers.load_state_dict(dense.memory_readers.state_dict())
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        for passes in (2, 3):
            expected = dense.compute_passes(ids, passes=passes).final.hidden_states
            actual = periodic.compute_passes(ids, passes=passes).final.hidden_states
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_periodic_write_mask_uses_completed_stride_positions():
    model = tape_model(mode="periodic", stride=4)
    ids = torch.arange(10)[None, :]
    assert model.write_mask(ids).tolist() == [[
        False, False, False, True, False, False, False, True, False, False
    ]]


def test_dense_write_mask_writes_every_position():
    model = tape_model(mode="dense", stride=1)
    ids = torch.arange(6)[None, :]
    assert model.write_mask(ids).all()


def test_periodic_write_is_strict_past():
    model = tape_model(mode="periodic", stride=4).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    embeddings = model.input_embeddings(ids)
    previous = torch.randn_like(embeddings)
    perturbed = previous.clone()
    perturbed[:, 3, :] += 3.0
    with torch.no_grad():
        base = model._run_feedback_hidden(ids, embeddings, previous)
        changed = model._run_feedback_hidden(ids, embeddings, perturbed)
    # The record written at position 3 cannot affect query position 3 itself.
    torch.testing.assert_close(changed[:, :4, :], base[:, :4, :], atol=0, rtol=0)
    assert not torch.allclose(changed[:, 4:, :], base[:, 4:, :])


def test_phase_a_trains_writer_and_readers_but_not_backbone():
    model = tape_model(mode="periodic")
    configure_phase(model, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = model.compute_loss(ids, phase="A", passes=2, loss_weights=[0.0, 1.0])
    output.loss.backward()
    assert model.writer.proj.weight.grad is not None
    assert model.writer.proj.weight.grad.abs().sum() > 0
    added = {id(parameter) for parameter in model.added_parameters()}
    for parameter in model.parameters():
        if id(parameter) not in added:
            assert parameter.grad is None


def test_seeded_periodic_tape_keeps_last_w_records():
    model = tape_model(mode="periodic", stride=2, window=3).eval()
    hidden = torch.arange(1 * 8 * model.config.hidden_size, dtype=torch.float32).reshape(
        1, 8, model.config.hidden_size
    )
    ids = torch.arange(8)[None, :]
    state = model._feedback_memory_from_hidden(hidden, input_ids=ids)
    assert isinstance(state, TapeState)
    assert state.valid.tolist() == [[True, True, True]]
    torch.testing.assert_close(state.memories[0], hidden[0, [3, 5, 7], :])


def test_memory_token_is_input_only_and_targets_skip_control_slot():
    model = tape_model(mode="memory_token").eval()
    V = model.config.vocab_size
    ids = torch.tensor([[4, V, 9, 11]])
    assert model.build_lm_labels(ids).tolist() == [[9, -100, 11, -100]]
    with torch.no_grad():
        out = model.compute_passes(ids, passes=2)
    assert out.final.logits.shape[-1] == V
    assert model.memory_token_id == V
    assert model.backbone.model.embed_tokens.num_embeddings == V


def test_memory_token_loss_has_zero_direct_gradient_at_control_position():
    model = tape_model(mode="memory_token").eval()
    V = model.config.vocab_size
    ids = torch.tensor([[4, V, 9, 11]])
    labels = model.build_lm_labels(ids)
    logits = torch.randn(1, 4, V, requires_grad=True)
    loss = causal_lm_loss_from_labels(logits, labels)
    loss.backward()

    # h(A) predicts B across the inserted control slot. The MEM hidden has no
    # direct LM objective at all, while B predicts C normally.
    assert labels.tolist() == [[9, -100, 11, -100]]
    assert logits.grad is not None
    assert logits.grad[0, 0].abs().sum() > 0
    torch.testing.assert_close(logits.grad[0, 1], torch.zeros_like(logits.grad[0, 1]), atol=0, rtol=0)
    assert logits.grad[0, 2].abs().sum() > 0
    torch.testing.assert_close(logits.grad[0, 3], torch.zeros_like(logits.grad[0, 3]), atol=0, rtol=0)


def test_memory_token_loss_is_invariant_to_control_position_logits():
    model = tape_model(mode="memory_token").eval()
    V = model.config.vocab_size
    ids = torch.tensor([[4, V, 9, 11]])
    labels = model.build_lm_labels(ids)
    logits = torch.randn(1, 4, V)
    changed = logits.clone()
    changed[:, 1, :] = 1_000.0 * torch.randn_like(changed[:, 1, :])
    changed[:, 3, :] = 1_000.0 * torch.randn_like(changed[:, 3, :])
    torch.testing.assert_close(
        causal_lm_loss_from_labels(logits, labels),
        causal_lm_loss_from_labels(changed, labels),
        atol=0,
        rtol=0,
    )


def test_memory_token_embedding_gets_phase_a_gradient_through_recurrence():
    model = tape_model(mode="memory_token", visibility="write_only")
    configure_phase(model, "A")
    V = model.config.vocab_size
    ids = torch.tensor([[1, 2, V, 3, 4, V, 5]])
    output = model.compute_loss(ids, phase="A", passes=2, loss_weights=[0.0, 1.0])
    output.loss.backward()
    assert model.memory_token_embedding is not None
    grad = model.memory_token_embedding.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    added = {id(parameter) for parameter in model.added_parameters()}
    for parameter in model.parameters():
        if id(parameter) not in added:
            assert parameter.grad is None


def test_write_only_mem_reads_context_but_is_not_visible_to_future_tokens():
    model = tape_model(mode="memory_token", visibility="write_only").eval()
    V = model.config.vocab_size
    ids = torch.tensor([[1, V, 3, 4]])
    with torch.no_grad():
        model.memory_token_embedding.zero_()
        base = model._run_first_hidden(ids)
        model.memory_token_embedding.fill_(0.75)
        changed = model._run_first_hidden(ids)
    assert not torch.allclose(base[:, 1, :], changed[:, 1, :])
    torch.testing.assert_close(base[:, 2:, :], changed[:, 2:, :], atol=0, rtol=0)

    ids2 = torch.tensor([[2, V, 3, 4]])
    with torch.no_grad():
        other_context = model._run_first_hidden(ids2)
    assert not torch.allclose(changed[:, 1, :], other_context[:, 1, :])


def test_visible_mem_can_affect_later_ordinary_states_locally():
    model = tape_model(mode="memory_token", visibility="visible").eval()
    V = model.config.vocab_size
    ids = torch.tensor([[1, V, 3, 4]])
    with torch.no_grad():
        model.memory_token_embedding.zero_()
        base = model._run_first_hidden(ids)
        model.memory_token_embedding.fill_(0.75)
        changed = model._run_first_hidden(ids)
    assert not torch.allclose(base[:, 2:, :], changed[:, 2:, :])


def test_mem_write_is_strict_past_for_tape_reader():
    model = tape_model(mode="memory_token").eval()
    V = model.config.vocab_size
    ids = torch.tensor([[1, V, 3, 4]])
    embeddings = model.input_embeddings(ids)
    previous = torch.randn_like(embeddings)
    perturbed = previous.clone(); perturbed[:, 1, :] += 5.0
    with torch.no_grad():
        base = model._run_feedback_hidden(ids, embeddings, previous)
        changed = model._run_feedback_hidden(ids, embeddings, perturbed)
    torch.testing.assert_close(base[:, :2, :], changed[:, :2, :], atol=0, rtol=0)
    assert not torch.allclose(base[:, 2:, :], changed[:, 2:, :])


def test_hybrid_mem_and_following_token_use_same_previous_ordinary_fast_source():
    model = tape_model(mode="memory_token", hybrid=True)
    V = model.config.vocab_size
    ids = torch.tensor([[10, V, 11, 12, V, 13]])
    dim = model.config.hidden_size
    hidden = torch.arange(ids.shape[1], dtype=torch.float32)[None, :, None].expand(1, -1, dim)
    aligned = model._previous_ordinary_hidden(hidden, ids)
    # source positions by physical position: none, A, A, B, C, C
    expected = torch.tensor([0, 0, 0, 2, 3, 3], dtype=torch.float32)
    torch.testing.assert_close(aligned[0, :, 0], expected, atol=0, rtol=0)


def test_hybrid_mem_writes_tape_without_advancing_fast_state():
    model = tape_model(mode="memory_token", hybrid=True).eval()
    V = model.config.vocab_size
    dim = model.config.hidden_size
    old_fast = torch.full((1, 1, dim), 2.0)
    state = HybridFeedbackState(
        fast_hidden=old_fast,
        tape=TapeState(
            memories=torch.zeros(1, model.memory_window, dim),
            valid=torch.zeros(1, model.memory_window, dtype=torch.bool),
        ),
    )
    mem_hidden = torch.full((1, 1, dim), 3.0)
    updated = model._append_feedback_memory(state, mem_hidden, token=torch.tensor([[V]]), position=1)
    torch.testing.assert_close(updated.fast_hidden, old_fast, atol=0, rtol=0)
    assert updated.tape.valid.tolist() == [[True, False, False, False]]

    ordinary_hidden = torch.full((1, 1, dim), 4.0)
    updated2 = model._append_feedback_memory(updated, ordinary_hidden, token=torch.tensor([[7]]), position=2)
    torch.testing.assert_close(updated2.fast_hidden, ordinary_hidden, atol=0, rtol=0)
    assert torch.equal(updated2.tape.valid, updated.tape.valid)


def test_memory_token_packing_length_and_order():
    ordinary = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    packed = insert_memory_tokens(ordinary, memory_token_id=99, interval=3)
    assert packed.tolist() == [[1, 2, 3, 99, 4, 5, 6, 99, 7]]
    assert memory_token_physical_length(7, 3) == 9


def test_invalid_memory_token_id_is_rejected_on_cpu():
    model = tape_model(mode="memory_token")
    V = model.config.vocab_size
    with pytest.raises(ValueError, match="outside"):
        model.input_embeddings(torch.tensor([[1, V + 1, 2]]))
