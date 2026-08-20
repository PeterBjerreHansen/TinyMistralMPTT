from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM, MistralRMSNorm

from ..feedback import HybridFeedbackState
from .multipass import shift_previous_hidden
from .tape import TapeVariant


class TapeAddHybridVariant(TapeVariant):
    """Tape recurrence plus the one-step MemoryAdd fast channel.

    In memory-token mode, ``<MEM>`` is a write computation rather than a
    linguistic-state transition.  The most recent *ordinary* previous-stream
    hidden therefore feeds both the MEM input and the following ordinary token;
    the MEM hidden writes the tape but never replaces the fast recurrent state.
    """

    variant_name = "tape_add_hybrid"

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        memory_window: int = 32,
        memory_write_mode: str = "periodic",
        memory_write_stride: int = 8,
        memory_token_visibility: str = "visible",
        initialization_seed: int = 4242,
    ):
        super().__init__(
            backbone,
            memory_window=memory_window,
            memory_write_mode=memory_write_mode,
            memory_write_stride=memory_write_stride,
            memory_token_visibility=memory_token_visibility,
            initialization_seed=initialization_seed,
        )
        hidden_size = int(backbone.config.hidden_size)
        self.memory_norm = MistralRMSNorm(hidden_size, eps=float(backbone.config.rms_norm_eps))
        with torch.random.fork_rng(devices=[]):
            self.memory_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.memory_projection.weight)

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from super().added_parameters()
        yield from self.memory_norm.parameters()
        yield from self.memory_projection.parameters()

    def _previous_ordinary_hidden(
        self,
        previous_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Strictly previous ordinary source for each physical model position.

        Example ``A <MEM> B`` maps both ``<MEM>`` and ``B`` to ``h_A``.  For
        periodic mode there are no control positions and this reduces exactly
        to the historical one-token right shift.
        """
        if previous_hidden.ndim != 3 or input_ids.shape != previous_hidden.shape[:2]:
            raise ValueError("previous_hidden/input_ids must align as [B,T,D]/[B,T]")
        if not self.uses_memory_tokens:
            return shift_previous_hidden(previous_hidden)
        ordinary = ~self.memory_token_mask(input_ids)
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        candidates = torch.where(
            ordinary,
            positions[None, :].expand(bsz, -1),
            torch.full((bsz, seq_len), -1, device=input_ids.device, dtype=torch.long),
        )
        inclusive = torch.cummax(candidates, dim=1).values
        strict = torch.cat(
            (
                torch.full((bsz, 1), -1, device=input_ids.device, dtype=torch.long),
                inclusive[:, :-1],
            ),
            dim=1,
        )
        safe = strict.clamp_min(0)
        gathered = previous_hidden.gather(
            1, safe[:, :, None].expand(-1, -1, previous_hidden.shape[-1])
        )
        return torch.where(strict[:, :, None] >= 0, gathered, torch.zeros_like(gathered))

    def memory_residual(
        self,
        previous_hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is None:
            if self.uses_memory_tokens:
                raise ValueError("memory-token hybrid residual requires input_ids")
            source = shift_previous_hidden(previous_hidden)
        else:
            source = self._previous_ordinary_hidden(previous_hidden, input_ids)
        return self.memory_projection(self.memory_norm(source))

    def _run_feedback_hidden_components(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        *,
        fast_hidden: torch.Tensor,
        tape_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if token_embeddings.shape != fast_hidden.shape or token_embeddings.shape != tape_hidden.shape:
            raise ValueError("token embeddings and both feedback sources must share [B,T,D]")
        feedback_inputs = token_embeddings + self.memory_residual(fast_hidden, input_ids)
        tape = self.build_tape(tape_hidden, input_ids)
        hidden, _ = self._run_tape_feedback_core(
            feedback_inputs,
            tape,
            past_key_values=None,
            use_cache=False,
            self_attention_mask=self.self_attention_key_mask(input_ids),
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
        feedback_inputs = token_embeddings + self.memory_residual(previous_hidden, input_ids)
        tape = self.build_tape(previous_hidden, input_ids)
        hidden, cache = self._run_tape_feedback_core(
            feedback_inputs,
            tape,
            past_key_values=None,
            use_cache=True,
            self_attention_mask=self.self_attention_key_mask(input_ids),
        )
        if cache is None:
            raise RuntimeError("cached TapeAddHybrid prefill did not return KV state")
        return hidden, cache

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: HybridFeedbackState,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        if not isinstance(feedback_memory, HybridFeedbackState):
            raise TypeError("TapeAddHybrid cached feedback requires HybridFeedbackState")
        if token_embedding.ndim != 3 or token_embedding.shape[1] != 1:
            raise ValueError("token_embedding must be [B,1,D]")
        if feedback_memory.fast_hidden.shape != token_embedding.shape:
            raise ValueError("hybrid fast feedback must match token embedding shape")
        if token is None:
            raise ValueError("TapeAddHybrid cached feedback requires current token ID")
        feedback_input = token_embedding + self.memory_projection(
            self.memory_norm(feedback_memory.fast_hidden)
        )
        hidden, cache = self._run_tape_feedback_core(
            feedback_input,
            feedback_memory.tape,
            past_key_values=past_key_values,
            use_cache=True,
            self_attention_mask=self.self_attention_key_mask(token),
        )
        if cache is None:
            raise RuntimeError("cached TapeAddHybrid token did not return KV state")
        return hidden, cache

    def _last_ordinary_hidden(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.uses_memory_tokens:
            return hidden_states[:, -1:, :].detach()
        if input_ids is None:
            raise ValueError("memory-token hybrid state requires input_ids")
        ordinary = ~self.memory_token_mask(input_ids)
        if bool((ordinary.sum(dim=1) == 0).any()):
            raise ValueError("each sequence needs at least one ordinary token for hybrid fast state")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        last = torch.where(ordinary, positions, torch.full_like(positions, -1)).max(dim=1).values
        gathered = hidden_states.gather(
            1, last[:, None, None].expand(-1, 1, hidden_states.shape[-1])
        )
        return gathered.detach()

    def _feedback_memory_from_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> HybridFeedbackState:
        tape = super()._feedback_memory_from_hidden(hidden_states, input_ids=input_ids)
        return HybridFeedbackState(
            fast_hidden=self._last_ordinary_hidden(hidden_states, input_ids),
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
            raise TypeError("TapeAddHybrid feedback requires HybridFeedbackState")
        trigger = self._write_trigger(token=token, position=position)
        tape = self._append_tape(feedback_memory.tape, new_hidden, trigger=trigger)
        if self.uses_memory_tokens:
            if token is None:
                raise ValueError("memory-token hybrid update requires current token")
            is_mem = self.memory_token_mask(token)[:, 0]
            # MEM writes the slow tape but does not advance the fast state.
            fast = torch.where(
                is_mem[:, None, None],
                feedback_memory.fast_hidden,
                new_hidden.detach(),
            )
        else:
            fast = new_hidden.detach()
        return HybridFeedbackState(fast_hidden=fast, tape=tape)
