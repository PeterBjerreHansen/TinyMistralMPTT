from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM

from ..feedback import SparseTapeState
from .memory_tape32 import MemoryTapeReader
from .multipass import MultiPassVariant


@dataclass(frozen=True)
class SparseTapeBatch:
    """Compact full-sequence sparse tape plus strict-past write counts."""

    memories: torch.Tensor  # [B,M,D], padded chronologically per example
    valid: torch.Tensor  # bool [B,M]
    writes_before: torch.Tensor  # [B,T]

    def __post_init__(self) -> None:
        if self.memories.ndim != 3:
            raise ValueError("SparseTapeBatch.memories must be [B,M,D]")
        if self.valid.shape != self.memories.shape[:2] or self.valid.dtype != torch.bool:
            raise ValueError("SparseTapeBatch.valid must be bool [B,M]")
        if self.writes_before.ndim != 2:
            raise ValueError("SparseTapeBatch.writes_before must be [B,T]")
        if self.writes_before.shape[0] != self.memories.shape[0]:
            raise ValueError("sparse tape batch sizes differ")


class SparseTapeWriter(nn.Module):
    """Minimal learnable write layer, identity-initialized for C=1 equivalence."""

    def __init__(self, hidden_size: int):
        super().__init__()
        # nn.Linear consumes the global RNG in its constructor even though this
        # writer is subsequently deterministic. Fork so architecture creation
        # cannot perturb experiment/data RNG state.
        with torch.random.fork_rng(devices=[]):
            self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
            with torch.no_grad():
                nn.init.eye_(self.proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("writer input must be [B,T,D]")
        return self.proj(hidden_states)


class SparseMemoryTapeVariant(MultiPassVariant):
    """MemoryTape reader with sparsely committed, learned memory records.

    The per-layer reader is exactly :class:`MemoryTapeReader`.  The only
    architectural change relative to dense MemoryTape32 is tape population:
    selected previous-pass top states pass through one identity-initialized
    linear writer and are committed to a compact tape.  Query position ``t``
    can read only records committed at positions ``< t``.
    """

    variant_name = "sparse_memory_tape"
    supports_cached_feedback = True

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
        super().__init__(backbone)
        if memory_window <= 0:
            raise ValueError("memory_window must be positive")
        if memory_write_mode not in {"periodic", "token"}:
            raise ValueError("memory_write_mode must be 'periodic' or 'token'")
        if memory_write_stride <= 0:
            raise ValueError("memory_write_stride must be positive")
        if memory_write_mode == "token" and memory_token_id is None:
            raise ValueError("token write mode requires memory_token_id")
        if memory_write_mode == "periodic" and memory_token_id is not None:
            raise ValueError("periodic write mode must not set memory_token_id")
        if memory_token_id is not None and not 0 <= int(memory_token_id) < int(backbone.config.vocab_size):
            raise ValueError("memory_token_id must lie inside the model vocabulary")

        self.memory_window = int(memory_window)
        self.memory_write_mode = str(memory_write_mode)
        self.memory_write_stride = int(memory_write_stride)
        self.memory_token_id = None if memory_token_id is None else int(memory_token_id)
        hidden_size = int(backbone.config.hidden_size)
        self.writer = SparseTapeWriter(hidden_size)
        self.memory_readers = nn.ModuleList(
            [
                MemoryTapeReader(
                    backbone,
                    window=self.memory_window,
                    initialization_seed=int(initialization_seed) + layer_index,
                )
                for layer_index in range(len(backbone.model.layers))
            ]
        )

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.writer.parameters()
        yield from self.memory_readers.parameters()

    def write_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if self.memory_write_mode == "periodic":
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            row = (positions + 1).remainder(self.memory_write_stride).eq(0)
            return row[None, :].expand(input_ids.shape[0], -1)
        assert self.memory_token_id is not None
        return input_ids.eq(self.memory_token_id)

    def _compact_written_states(
        self,
        written_states: torch.Tensor,
        write_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compact selected states chronologically, padding only across batch."""
        if written_states.ndim != 3 or write_mask.shape != written_states.shape[:2]:
            raise ValueError("written_states/write_mask shapes are incompatible")
        counts = write_mask.sum(dim=1)
        max_count = int(counts.max().item()) if counts.numel() else 0
        rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for batch_index in range(written_states.shape[0]):
            selected = written_states[batch_index, write_mask[batch_index], :]
            count = selected.shape[0]
            if count < max_count:
                padding = selected.new_zeros((max_count - count, selected.shape[-1]))
                selected = torch.cat((selected, padding), dim=0)
            rows.append(selected)
            valid = torch.arange(max_count, device=write_mask.device) < count
            masks.append(valid)
        if not rows:
            return (
                written_states.new_zeros((0, 0, written_states.shape[-1])),
                write_mask.new_zeros((0, 0)),
            )
        return torch.stack(rows, dim=0), torch.stack(masks, dim=0)

    def build_sparse_tape(
        self,
        previous_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> SparseTapeBatch:
        if previous_hidden.ndim != 3 or input_ids.ndim != 2:
            raise ValueError("previous_hidden must be [B,T,D] and input_ids [B,T]")
        if previous_hidden.shape[:2] != input_ids.shape:
            raise ValueError("previous_hidden and input_ids token shapes differ")
        mask = self.write_mask(input_ids)
        # Compact first so the learned D->D writer is evaluated only at write
        # events (plus unavoidable cross-batch padding), preserving the sparse
        # write-compute advantage instead of projecting every source token.
        selected, valid = self._compact_written_states(previous_hidden, mask)
        memories = self.writer(selected)
        cumulative = mask.long().cumsum(dim=1)
        writes_before = cumulative - mask.long()
        return SparseTapeBatch(
            memories=memories,
            valid=valid,
            writes_before=writes_before,
        )

    @staticmethod
    def _cache_next_position(past_key_values: tuple[LayerKVCache, ...]) -> int:
        if not past_key_values:
            raise ValueError("past_key_values must not be empty")
        positions = {cache.next_position for cache in past_key_values}
        if len(positions) != 1:
            raise ValueError("layer caches disagree on next absolute position")
        return next(iter(positions))

    def _run_sparse_feedback_core(
        self,
        token_embeddings: torch.Tensor,
        tape: SparseTapeBatch | SparseTapeState,
        *,
        past_key_values: tuple[LayerKVCache, ...] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...] | None]:
        if token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must be [B,T,D]")
        if token_embeddings.shape[0] != tape.memories.shape[0]:
            raise ValueError("token and tape batch sizes differ")
        if token_embeddings.shape[-1] != tape.memories.shape[-1]:
            raise ValueError("token and tape hidden dimensions differ")
        cached_bank = isinstance(tape, SparseTapeState)
        if cached_bank and token_embeddings.shape[1] != 1:
            raise ValueError("cached sparse-tape query must contain exactly one token")
        if cached_bank and tape.capacity != self.memory_window:
            raise ValueError("cached sparse-tape capacity differs from memory_window")
        if past_key_values is not None and len(past_key_values) != len(self.backbone.model.layers):
            raise ValueError("past_key_values must contain one cache per layer")

        bsz, seq_len, _ = token_embeddings.shape
        start = 0 if past_key_values is None else self._cache_next_position(past_key_values)
        position_ids = torch.arange(
            start,
            start + seq_len,
            device=token_embeddings.device,
            dtype=torch.long,
        )[None, :].expand(bsz, -1)

        hidden_states = token_embeddings
        new_caches: list[LayerKVCache] | None = [] if use_cache else None
        for layer_index, (layer, memory_reader) in enumerate(
            zip(self.backbone.model.layers, self.memory_readers, strict=True)
        ):
            residual = hidden_states
            x = layer.input_layernorm(hidden_states)
            past = None if past_key_values is None else past_key_values[layer_index]
            x, cache = layer.self_attn(
                x,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past,
                use_cache=use_cache,
                fast_attention_compatible=past_key_values is None,
            )
            hidden_states = residual + x
            if new_caches is not None:
                if cache is None:
                    raise RuntimeError("cached sparse-tape layer did not return KV state")
                new_caches.append(cache)

            if cached_bank:
                memory_delta = memory_reader.forward_bank(
                    hidden_states,
                    tape.memories,
                    memory_mask=tape.valid,
                )
            else:
                assert isinstance(tape, SparseTapeBatch)
                memory_delta = memory_reader.forward_sparse(
                    hidden_states,
                    tape.memories,
                    writes_before=tape.writes_before,
                    memory_mask=tape.valid,
                )
            hidden_states = hidden_states + memory_delta

            residual = hidden_states
            x = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(x)

        hidden_states = self.backbone.model.norm(hidden_states)
        return hidden_states, tuple(new_caches) if new_caches is not None else None

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        tape = self.build_sparse_tape(previous_hidden, input_ids)
        hidden, _ = self._run_sparse_feedback_core(
            token_embeddings,
            tape,
            past_key_values=None,
            use_cache=False,
        )
        return hidden

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        tape = self.build_sparse_tape(previous_hidden, input_ids)
        hidden, cache = self._run_sparse_feedback_core(
            token_embeddings,
            tape,
            past_key_values=None,
            use_cache=True,
        )
        if cache is None:
            raise RuntimeError("cached SparseMemoryTape prefill did not return KV state")
        return hidden, cache

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: SparseTapeState,
        past_key_values: tuple[LayerKVCache, ...],
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        if not isinstance(feedback_memory, SparseTapeState):
            raise TypeError("SparseMemoryTape cached feedback requires SparseTapeState")
        hidden, cache = self._run_sparse_feedback_core(
            token_embedding,
            feedback_memory,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if cache is None:
            raise RuntimeError("cached SparseMemoryTape token did not return KV state")
        return hidden, cache

    def _state_from_tape_batch(self, tape: SparseTapeBatch) -> SparseTapeState:
        bsz, _, dim = tape.memories.shape
        result = tape.memories.new_zeros((bsz, self.memory_window, dim))
        valid = torch.zeros(
            (bsz, self.memory_window), dtype=torch.bool, device=tape.memories.device
        )
        for batch_index in range(bsz):
            row = tape.memories[batch_index, tape.valid[batch_index], :]
            row = row[-self.memory_window :]
            count = row.shape[0]
            if count:
                result[batch_index, :count, :] = row
                valid[batch_index, :count] = True
        return SparseTapeState(result.detach(), valid)

    def _feedback_memory_from_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> SparseTapeState:
        if hidden_states.ndim != 3 or hidden_states.shape[1] < 1:
            raise ValueError("hidden_states must be non-empty [B,T,D]")
        if input_ids is None:
            if self.memory_write_mode == "token":
                raise ValueError("token write mode requires input_ids to seed feedback memory")
            input_ids = torch.zeros(
                hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device
            )
        tape = self.build_sparse_tape(hidden_states, input_ids)
        return self._state_from_tape_batch(tape)

    def _write_trigger(
        self,
        *,
        token: torch.Tensor | None,
        position: int | None,
    ) -> torch.Tensor:
        if self.memory_write_mode == "periodic":
            if position is None:
                raise ValueError("periodic cached write requires absolute position")
            if token is None:
                raise ValueError("cached write requires token for batch size")
            trigger = (int(position) + 1) % self.memory_write_stride == 0
            return torch.full(
                (token.shape[0],), trigger, dtype=torch.bool, device=token.device
            )
        if token is None:
            raise ValueError("token cached write requires current token")
        if token.ndim != 2 or token.shape[1] != 1:
            raise ValueError("cached token must be [B,1]")
        assert self.memory_token_id is not None
        return token[:, 0].eq(self.memory_token_id)

    def _append_sparse_tape(
        self,
        state: SparseTapeState,
        new_hidden: torch.Tensor,
        *,
        trigger: torch.Tensor,
    ) -> SparseTapeState:
        if new_hidden.ndim != 3 or new_hidden.shape[1] != 1:
            raise ValueError("new_hidden must be [B,1,D]")
        if trigger.shape != (new_hidden.shape[0],) or trigger.dtype != torch.bool:
            raise ValueError("trigger must be bool [B]")
        if state.batch_size != new_hidden.shape[0] or state.hidden_size != new_hidden.shape[-1]:
            raise ValueError("sparse tape and new hidden shapes are incompatible")
        new_record = self.writer(new_hidden).detach()
        memories = state.memories.clone()
        valid = state.valid.clone()
        counts = valid.sum(dim=1, dtype=torch.long)
        full_trigger = trigger & counts.eq(self.memory_window)

        # Full rows drop their oldest record. Partial rows keep their existing
        # left-aligned records and write into the next free slot. Every tensor
        # operation remains device-side, avoiding a host synchronization for
        # each batch example during recurrent decoding.
        shifted_memories = torch.cat(
            (memories[:, 1:, :], torch.zeros_like(memories[:, :1, :])), dim=1
        )
        shifted_valid = torch.cat(
            (valid[:, 1:], torch.zeros_like(valid[:, :1])), dim=1
        )
        memories = torch.where(full_trigger[:, None, None], shifted_memories, memories)
        valid = torch.where(full_trigger[:, None], shifted_valid, valid)

        write_index = counts.clamp(max=self.memory_window - 1)
        scatter_index = write_index[:, None, None].expand(-1, 1, new_record.shape[-1])
        candidate_memories = memories.scatter(1, scatter_index, new_record)
        candidate_valid = valid.scatter(
            1, write_index[:, None], torch.ones_like(trigger[:, None])
        )
        memories = torch.where(trigger[:, None, None], candidate_memories, memories)
        valid = torch.where(trigger[:, None], candidate_valid, valid)
        return SparseTapeState(memories.detach(), valid)

    def _append_feedback_memory(
        self,
        feedback_memory: SparseTapeState,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> SparseTapeState:
        if not isinstance(feedback_memory, SparseTapeState):
            raise TypeError("SparseMemoryTape feedback requires SparseTapeState")
        trigger = self._write_trigger(token=token, position=position)
        return self._append_sparse_tape(feedback_memory, new_hidden, trigger=trigger)
