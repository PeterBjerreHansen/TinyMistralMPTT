from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM, MistralRMSNorm

from ..feedback import HybridFeedbackState
from .multipass import shift_previous_hidden
from .sparse_memory_tape import SparseMemoryTapeVariant


class MemoryAddSparseTapeVariant(SparseMemoryTapeVariant):
    """Simple hybrid: immediate MemoryAdd recurrence plus SparseMemoryTape.

    There is deliberately no gate/controller between the two channels.  Every
    feedback pass adds the existing one-token MemoryAdd residual to the input
    embedding and independently uses the same sparse tape reader implemented by
    :class:`SparseMemoryTapeVariant`.
    """

    variant_name = "memory_add_sparse_tape"

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        memory_window: int = 32,
        memory_write_mode: str = "periodic",
        memory_write_stride: int = 8,
        memory_token_id: int | None = None,
        initialization_seed: int = 4242,
    ):
        super().__init__(
            backbone,
            memory_window=memory_window,
            memory_write_mode=memory_write_mode,
            memory_write_stride=memory_write_stride,
            memory_token_id=memory_token_id,
            initialization_seed=initialization_seed,
        )
        hidden_size = int(backbone.config.hidden_size)
        self.memory_norm = MistralRMSNorm(
            hidden_size, eps=float(backbone.config.rms_norm_eps)
        )
        with torch.random.fork_rng(devices=[]):
            self.memory_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.memory_projection.weight)

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from super().added_parameters()
        yield from self.memory_norm.parameters()
        yield from self.memory_projection.parameters()

    def memory_residual(self, previous_hidden: torch.Tensor) -> torch.Tensor:
        shifted = shift_previous_hidden(previous_hidden)
        return self.memory_projection(self.memory_norm(shifted))

    def _run_feedback_hidden_components(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        *,
        fast_hidden: torch.Tensor,
        tape_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Run one feedback pass with independently supplied fast/tape sources.

        This is intentionally a thin diagnostic hook: normal model execution
        passes the same previous-pass hidden sequence to both channels, while
        memory-intervention tools can zero or mismatch them independently.
        """
        if token_embeddings.shape != fast_hidden.shape or token_embeddings.shape != tape_hidden.shape:
            raise ValueError("token embeddings and both feedback sources must share [B,T,D]")
        feedback_inputs = token_embeddings + self.memory_residual(fast_hidden)
        tape = self.build_sparse_tape(tape_hidden, input_ids)
        hidden, _ = self._run_sparse_feedback_core(
            feedback_inputs,
            tape,
            past_key_values=None,
            use_cache=False,
        )
        return hidden

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        return self._run_feedback_hidden_components(
            input_ids,
            token_embeddings,
            fast_hidden=previous_hidden,
            tape_hidden=previous_hidden,
        )

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        if token_embeddings.shape != previous_hidden.shape:
            raise ValueError("token embeddings and previous hidden must share [B,T,D]")
        feedback_inputs = token_embeddings + self.memory_residual(previous_hidden)
        tape = self.build_sparse_tape(previous_hidden, input_ids)
        hidden, cache = self._run_sparse_feedback_core(
            feedback_inputs,
            tape,
            past_key_values=None,
            use_cache=True,
        )
        if cache is None:
            raise RuntimeError("cached hybrid prefill did not return KV state")
        return hidden, cache

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: HybridFeedbackState,
        past_key_values: tuple[LayerKVCache, ...],
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        if not isinstance(feedback_memory, HybridFeedbackState):
            raise TypeError("hybrid cached feedback requires HybridFeedbackState")
        if token_embedding.ndim != 3 or token_embedding.shape[1] != 1:
            raise ValueError("token_embedding must be [B,1,D]")
        if feedback_memory.fast_hidden.shape != token_embedding.shape:
            raise ValueError("hybrid fast feedback must match token embedding shape")
        feedback_input = token_embedding + self.memory_projection(
            self.memory_norm(feedback_memory.fast_hidden)
        )
        hidden, cache = self._run_sparse_feedback_core(
            feedback_input,
            feedback_memory.tape,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if cache is None:
            raise RuntimeError("cached hybrid token did not return KV state")
        return hidden, cache

    def _feedback_memory_from_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> HybridFeedbackState:
        tape = super()._feedback_memory_from_hidden(hidden_states, input_ids=input_ids)
        return HybridFeedbackState(
            fast_hidden=hidden_states[:, -1:, :].detach(),
            tape=tape,
        )

    def _append_feedback_memory(
        self,
        feedback_memory: HybridFeedbackState,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> HybridFeedbackState:
        if not isinstance(feedback_memory, HybridFeedbackState):
            raise TypeError("hybrid feedback requires HybridFeedbackState")
        trigger = self._write_trigger(token=token, position=position)
        tape = self._append_sparse_tape(
            feedback_memory.tape,
            new_hidden,
            trigger=trigger,
        )
        return HybridFeedbackState(
            fast_hidden=new_hidden.detach(),
            tape=tape,
        )
