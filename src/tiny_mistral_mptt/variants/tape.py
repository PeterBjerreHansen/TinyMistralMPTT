from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM, MistralRMSNorm

from ..attention.memory_local import (
    memory_bank_attention,
    strict_past_local_attention,
    strict_past_tape_attention,
)
from ..feedback import TapeState
from .multipass import MultiPassVariant


MEMORY_WRITE_MODES = {"dense", "periodic", "memory_token"}
MEMORY_TOKEN_VISIBILITIES = {"visible", "write_only"}


class TapeReader(nn.Module):
    """Mistral-shaped GQA cross-attention into a strict-past local memory tape."""

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        window: int,
        initialization_seed: int,
    ):
        super().__init__()
        config = backbone.config
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.window = int(window)
        self.dropout_p = float(config.attention_dropout)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.query_norm = MistralRMSNorm(
                self.hidden_size, eps=config.rms_norm_eps
            )
            self.memory_norm = MistralRMSNorm(
                self.hidden_size, eps=config.rms_norm_eps
            )
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=False
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
            )
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim, self.hidden_size, bias=False
            )
            std = float(config.initializer_range)
            for module in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def _project(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3 or memory_states.ndim != 3:
            raise ValueError("hidden_states and memory_states must be [B,T,D]")
        if hidden_states.shape[0] != memory_states.shape[0]:
            raise ValueError("hidden and memory batch sizes differ")
        if (
            hidden_states.shape[-1] != self.hidden_size
            or memory_states.shape[-1] != self.hidden_size
        ):
            raise ValueError("hidden and memory dimensions differ from reader hidden size")

        bsz, query_len, _ = hidden_states.shape
        memory_len = memory_states.shape[1]
        query = self.q_proj(self.query_norm(hidden_states))
        memory = self.memory_norm(memory_states)
        key = self.k_proj(memory)
        value = self.v_proj(memory)

        query = query.view(
            bsz, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            bsz, memory_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            bsz, memory_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        return query, key, value

    def forward(
        self, hidden_states: torch.Tensor, memory_states: torch.Tensor
    ) -> torch.Tensor:
        if hidden_states.shape != memory_states.shape:
            raise ValueError("hidden_states and memory_states must share [B,T,D]")
        bsz, seq_len, _ = hidden_states.shape
        query, key, value = self._project(hidden_states, memory_states)
        output = strict_past_local_attention(
            query,
            key,
            value,
            window=self.window,
            dropout_p=self.dropout_p,
            training=self.training,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(output)

    def forward_bank(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend to a bank whose entries are already strictly in the past."""
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("cached Tape query must be [B,1,D]")
        if memory_states.shape[1] > self.window:
            raise ValueError("cached memory bank exceeds configured window")
        bsz, query_len, _ = hidden_states.shape
        query, key, value = self._project(hidden_states, memory_states)
        output = memory_bank_attention(
            query,
            key,
            value,
            memory_mask=memory_mask,
            dropout_p=self.dropout_p,
            training=self.training,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, query_len, -1)
        return self.o_proj(output)

    def forward_tape(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        *,
        writes_before: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attend to the last ``window`` committed records at each query token."""
        if hidden_states.ndim != 3 or memory_states.ndim != 3:
            raise ValueError("hidden_states and memory_states must be [B,T,D]")
        bsz, query_len, _ = hidden_states.shape
        query, key, value = self._project(hidden_states, memory_states)
        output = strict_past_tape_attention(
            query,
            key,
            value,
            writes_before=writes_before,
            memory_mask=memory_mask,
            window=self.window,
            dropout_p=self.dropout_p,
            training=self.training,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, query_len, -1)
        return self.o_proj(output)



@dataclass(frozen=True)
class TapeBatch:
    """Compact full-sequence tape plus strict-past write counts."""

    memories: torch.Tensor  # [B,M,D], padded chronologically per example
    valid: torch.Tensor  # bool [B,M]
    writes_before: torch.Tensor  # [B,T]

    def __post_init__(self) -> None:
        if self.memories.ndim != 3:
            raise ValueError("TapeBatch.memories must be [B,M,D]")
        if self.valid.shape != self.memories.shape[:2] or self.valid.dtype != torch.bool:
            raise ValueError("TapeBatch.valid must be bool [B,M]")
        if self.writes_before.ndim != 2:
            raise ValueError("TapeBatch.writes_before must be [B,T]")
        if self.writes_before.shape[0] != self.memories.shape[0]:
            raise ValueError("tape batch sizes differ")


class TapeWriter(nn.Module):
    """Minimal learned D->D storage transform, identity-initialized."""

    def __init__(self, hidden_size: int):
        super().__init__()
        # Linear's constructor consumes RNG before overwrite. Forking prevents
        # architecture construction from perturbing experiment/data RNG state.
        with torch.random.fork_rng(devices=[]):
            self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
            with torch.no_grad():
                nn.init.eye_(self.proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("writer input must be [B,T,D]")
        return self.proj(hidden_states)


class TapeVariant(MultiPassVariant):
    """Learned tape memory with dense, periodic, or explicit-memory-token writes.

    Dense mode writes every top state; periodic mode writes selected ordinary-token top states. Memory-token mode
    treats ID ``backbone.config.vocab_size`` as an input-only ``<MEM>`` control
    position with its own learned embedding; that ID is never an LM output
    class. A MEM state predicts nothing and writes exactly one tape record.
    """

    variant_name = "tape"
    supports_cached_feedback = True

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
        super().__init__(backbone)
        if memory_window <= 0:
            raise ValueError("memory_window must be positive")
        if memory_write_mode not in MEMORY_WRITE_MODES:
            raise ValueError("memory_write_mode must be 'dense', 'periodic', or 'memory_token'")
        if memory_write_stride <= 0:
            raise ValueError("memory_write_stride must be positive")
        if memory_token_visibility not in MEMORY_TOKEN_VISIBILITIES:
            raise ValueError("memory_token_visibility must be 'visible' or 'write_only'")
        if memory_write_mode != "memory_token" and memory_token_visibility != "visible":
            raise ValueError("memory_token_visibility applies only to memory_token mode")

        base_vocab = int(backbone.config.vocab_size)
        self.memory_window = int(memory_window)
        self.memory_write_mode = str(memory_write_mode)
        self.memory_write_stride = int(memory_write_stride)
        self.memory_token_visibility = str(memory_token_visibility)
        self.memory_token_id = base_vocab if memory_write_mode == "memory_token" else None
        self.base_vocab_size = base_vocab

        hidden_size = int(backbone.config.hidden_size)
        self.writer = TapeWriter(hidden_size)
        if self.memory_write_mode == "memory_token":
            # Zero-init is deliberately conservative: the control slot begins
            # without adding lexical content but can contextualize through the
            # transformer's attention and learns in Phase A as an added param.
            self.memory_token_embedding = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("memory_token_embedding", None)
        self.memory_readers = nn.ModuleList(
            [
                TapeReader(
                    backbone,
                    window=self.memory_window,
                    initialization_seed=int(initialization_seed) + layer_index,
                )
                for layer_index in range(len(backbone.model.layers))
            ]
        )

    @property
    def uses_memory_tokens(self) -> bool:
        return self.memory_write_mode == "memory_token"

    def phase_a_first_pass_requires_grad(self) -> bool:
        return self.uses_memory_tokens

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.writer.parameters()
        if self.memory_token_embedding is not None:
            yield self.memory_token_embedding
        yield from self.memory_readers.parameters()

    def memory_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if not self.uses_memory_tokens:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        assert self.memory_token_id is not None
        return input_ids.eq(self.memory_token_id)

    def control_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.memory_token_mask(input_ids)

    def prediction_hidden_after_sequence(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("hidden_states/input_ids token shapes differ")
        if not self.uses_memory_tokens:
            return hidden_states[:, -1:, :]
        ordinary = ~self.memory_token_mask(input_ids)
        if bool((ordinary.sum(dim=1) == 0).any()):
            raise ValueError("memory-token sequence has no linguistic position")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        last = torch.where(ordinary, positions, torch.full_like(positions, -1)).max(dim=1).values
        return hidden_states.gather(
            1, last[:, None, None].expand(-1, 1, hidden_states.shape[-1])
        )

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        # Avoid a host synchronization in every cached CUDA/MPS token step. The
        # checked data/config pipeline establishes the range before device
        # execution; CPU/direct-call paths retain eager, friendly validation.
        if input_ids.device.type == "cpu":
            if bool((input_ids < 0).any()):
                raise ValueError("input IDs must be non-negative")
            upper = self.base_vocab_size + (1 if self.uses_memory_tokens else 0)
            if bool((input_ids >= upper).any()):
                raise ValueError("input ID lies outside this variant's input vocabulary")
            if not self.uses_memory_tokens and bool((input_ids >= self.base_vocab_size).any()):
                raise ValueError("periodic tape variants accept only ordinary vocabulary IDs")

    def input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_input_ids(input_ids)
        if not self.uses_memory_tokens:
            return self.backbone.model.embed_tokens(input_ids)
        assert self.memory_token_id is not None and self.memory_token_embedding is not None
        is_mem = input_ids.eq(self.memory_token_id)
        safe_ids = input_ids.masked_fill(is_mem, 0)
        ordinary = self.backbone.model.embed_tokens(safe_ids)
        mem = self.memory_token_embedding.to(dtype=ordinary.dtype)[None, None, :]
        return torch.where(is_mem[:, :, None], mem, ordinary)

    def self_attention_key_mask(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if not self.uses_memory_tokens or self.memory_token_visibility == "visible":
            return None
        # Asymmetric write-only semantics: MEM remains a query and can read its
        # causal prefix, but its K/V is unavailable to every later query.
        return ~self.memory_token_mask(input_ids)

    def build_lm_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.uses_memory_tokens:
            return super().build_lm_labels(input_ids)
        self._validate_input_ids(input_ids)
        is_mem = self.memory_token_mask(input_ids)
        ordinary = ~is_mem
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        sentinel = torch.full((bsz, seq_len), seq_len, device=input_ids.device, dtype=torch.long)
        candidates = torch.where(ordinary, positions[None, :].expand(bsz, -1), sentinel)
        # For each physical position, find the nearest ordinary position strictly
        # to its right. This stays device-side; the former Python reverse scan
        # synchronized once per token on CUDA/MPS.
        suffix_min = torch.flip(
            torch.cummin(torch.flip(candidates, dims=(1,)), dim=1).values,
            dims=(1,),
        )
        next_index = torch.cat(
            (suffix_min[:, 1:], torch.full((bsz, 1), seq_len, device=input_ids.device, dtype=torch.long)),
            dim=1,
        )
        safe_index = next_index.clamp(max=max(seq_len - 1, 0))
        next_token = input_ids.gather(1, safe_index)
        valid = ordinary & next_index.lt(seq_len)
        return torch.where(valid, next_token, torch.full_like(input_ids, -100))

    def write_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_input_ids(input_ids)
        if self.memory_write_mode == "dense":
            return torch.ones_like(input_ids, dtype=torch.bool)
        if self.memory_write_mode == "periodic":
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            row = (positions + 1).remainder(self.memory_write_stride).eq(0)
            return row[None, :].expand(input_ids.shape[0], -1)
        return self.memory_token_mask(input_ids)

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
            masks.append(torch.arange(max_count, device=write_mask.device) < count)
        if not rows:
            return (
                written_states.new_zeros((0, 0, written_states.shape[-1])),
                write_mask.new_zeros((0, 0)),
            )
        return torch.stack(rows, dim=0), torch.stack(masks, dim=0)

    def build_tape(
        self,
        previous_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> TapeBatch:
        if previous_hidden.ndim != 3 or input_ids.ndim != 2:
            raise ValueError("previous_hidden must be [B,T,D] and input_ids [B,T]")
        if previous_hidden.shape[:2] != input_ids.shape:
            raise ValueError("previous_hidden and input_ids token shapes differ")
        mask = self.write_mask(input_ids)
        selected, valid = self._compact_written_states(previous_hidden, mask)
        memories = self.writer(selected)
        cumulative = mask.long().cumsum(dim=1)
        writes_before = cumulative - mask.long()
        return TapeBatch(memories=memories, valid=valid, writes_before=writes_before)

    @staticmethod
    def _cache_next_position(past_key_values: tuple[LayerKVCache, ...]) -> int:
        if not past_key_values:
            raise ValueError("past_key_values must not be empty")
        positions = {cache.next_position for cache in past_key_values}
        if len(positions) != 1:
            raise ValueError("layer caches disagree on next absolute position")
        return next(iter(positions))

    def _run_tape_feedback_core(
        self,
        token_embeddings: torch.Tensor,
        tape: TapeBatch | TapeState,
        *,
        past_key_values: tuple[LayerKVCache, ...] | None,
        use_cache: bool,
        self_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...] | None]:
        if token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must be [B,T,D]")
        if token_embeddings.shape[0] != tape.memories.shape[0]:
            raise ValueError("token and tape batch sizes differ")
        if token_embeddings.shape[-1] != tape.memories.shape[-1]:
            raise ValueError("token and tape hidden dimensions differ")
        cached_bank = isinstance(tape, TapeState)
        if cached_bank and token_embeddings.shape[1] != 1:
            raise ValueError("cached tape query must contain exactly one token")
        if cached_bank and tape.capacity != self.memory_window:
            raise ValueError("cached tape capacity differs from memory_window")
        if past_key_values is not None and len(past_key_values) != len(self.backbone.model.layers):
            raise ValueError("past_key_values must contain one cache per layer")

        bsz, seq_len, _ = token_embeddings.shape
        start = 0 if past_key_values is None else self._cache_next_position(past_key_values)
        position_ids = torch.arange(
            start, start + seq_len, device=token_embeddings.device, dtype=torch.long
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
                attention_mask=self_attention_mask,
                position_ids=position_ids,
                past_key_value=past,
                use_cache=use_cache,
                fast_attention_compatible=(past_key_values is None),
            )
            hidden_states = residual + x
            if new_caches is not None:
                if cache is None:
                    raise RuntimeError("cached tape layer did not return KV state")
                new_caches.append(cache)

            if cached_bank:
                memory_delta = memory_reader.forward_bank(
                    hidden_states, tape.memories, memory_mask=tape.valid
                )
            else:
                assert isinstance(tape, TapeBatch)
                memory_delta = memory_reader.forward_tape(
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
        tape = self.build_tape(previous_hidden, input_ids)
        hidden, _ = self._run_tape_feedback_core(
            token_embeddings,
            tape,
            past_key_values=None,
            use_cache=False,
            self_attention_mask=self.self_attention_key_mask(input_ids),
        )
        return hidden

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        tape = self.build_tape(previous_hidden, input_ids)
        hidden, cache = self._run_tape_feedback_core(
            token_embeddings,
            tape,
            past_key_values=None,
            use_cache=True,
            self_attention_mask=self.self_attention_key_mask(input_ids),
        )
        if cache is None:
            raise RuntimeError("cached Tape prefill did not return KV state")
        return hidden, cache

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: TapeState,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        if not isinstance(feedback_memory, TapeState):
            raise TypeError("Tape cached feedback requires TapeState")
        if token is None:
            raise ValueError("cached Tape requires the current token ID")
        hidden, cache = self._run_tape_feedback_core(
            token_embedding,
            feedback_memory,
            past_key_values=past_key_values,
            use_cache=True,
            self_attention_mask=self.self_attention_key_mask(token),
        )
        if cache is None:
            raise RuntimeError("cached Tape token did not return KV state")
        return hidden, cache

    def _state_from_tape_batch(self, tape: TapeBatch) -> TapeState:
        bsz, _, dim = tape.memories.shape
        result = tape.memories.new_zeros((bsz, self.memory_window, dim))
        valid = torch.zeros((bsz, self.memory_window), dtype=torch.bool, device=tape.memories.device)
        for batch_index in range(bsz):
            row = tape.memories[batch_index, tape.valid[batch_index], :][-self.memory_window :]
            count = row.shape[0]
            if count:
                result[batch_index, :count, :] = row
                valid[batch_index, :count] = True
        return TapeState(result.detach(), valid)

    def _feedback_memory_from_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> TapeState:
        if hidden_states.ndim != 3 or hidden_states.shape[1] < 1:
            raise ValueError("hidden_states must be non-empty [B,T,D]")
        if input_ids is None:
            if self.memory_write_mode == "memory_token":
                raise ValueError("memory-token mode requires input_ids to seed feedback memory")
            input_ids = torch.zeros(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)
        return self._state_from_tape_batch(self.build_tape(hidden_states, input_ids))

    def _write_trigger(
        self,
        *,
        token: torch.Tensor | None,
        position: int | None,
    ) -> torch.Tensor:
        if token is None or token.ndim != 2 or token.shape[1] != 1:
            raise ValueError("cached write requires token [B,1]")
        if self.memory_write_mode == "dense":
            return torch.ones((token.shape[0],), dtype=torch.bool, device=token.device)
        if self.memory_write_mode == "periodic":
            if position is None:
                raise ValueError("periodic cached write requires absolute position")
            trigger = (int(position) + 1) % self.memory_write_stride == 0
            return torch.full((token.shape[0],), trigger, dtype=torch.bool, device=token.device)
        assert self.memory_token_id is not None
        return token[:, 0].eq(self.memory_token_id)

    def _append_tape(
        self,
        state: TapeState,
        new_hidden: torch.Tensor,
        *,
        trigger: torch.Tensor,
    ) -> TapeState:
        if new_hidden.ndim != 3 or new_hidden.shape[1] != 1:
            raise ValueError("new_hidden must be [B,1,D]")
        if trigger.shape != (new_hidden.shape[0],) or trigger.dtype != torch.bool:
            raise ValueError("trigger must be bool [B]")
        if state.batch_size != new_hidden.shape[0] or state.hidden_size != new_hidden.shape[-1]:
            raise ValueError("tape and new hidden shapes are incompatible")
        new_record = self.writer(new_hidden).detach()
        memories = state.memories.clone()
        valid = state.valid.clone()
        counts = valid.sum(dim=1, dtype=torch.long)
        full_trigger = trigger & counts.eq(self.memory_window)
        shifted_memories = torch.cat(
            (memories[:, 1:, :], torch.zeros_like(memories[:, :1, :])), dim=1
        )
        shifted_valid = torch.cat((valid[:, 1:], torch.zeros_like(valid[:, :1])), dim=1)
        memories = torch.where(full_trigger[:, None, None], shifted_memories, memories)
        valid = torch.where(full_trigger[:, None], shifted_valid, valid)
        write_index = counts.clamp(max=self.memory_window - 1)
        scatter_index = write_index[:, None, None].expand(-1, 1, new_record.shape[-1])
        candidate_memories = memories.scatter(1, scatter_index, new_record)
        candidate_valid = valid.scatter(1, write_index[:, None], torch.ones_like(trigger[:, None]))
        memories = torch.where(trigger[:, None, None], candidate_memories, memories)
        valid = torch.where(trigger[:, None], candidate_valid, valid)
        return TapeState(memories.detach(), valid)

    def _append_feedback_memory(
        self,
        feedback_memory: TapeState,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> TapeState:
        if not isinstance(feedback_memory, TapeState):
            raise TypeError("Tape feedback requires TapeState")
        trigger = self._write_trigger(token=token, position=position)
        return self._append_tape(feedback_memory, new_hidden, trigger=trigger)
