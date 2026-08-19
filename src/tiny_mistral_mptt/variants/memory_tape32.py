from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM, MistralRMSNorm

from ..attention.memory_local import (
    memory_bank_attention,
    strict_past_local_attention,
    strict_past_sparse_memory_attention,
)
from .multipass import MultiPassVariant


class MemoryTapeReader(nn.Module):
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
            raise ValueError("cached MemoryTape32 query must be [B,1,D]")
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

    def forward_sparse(
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
        output = strict_past_sparse_memory_attention(
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


class MemoryTape32Variant(MultiPassVariant):
    """Per-layer cross-attention to the previous pass's last 32 deep states."""

    variant_name = "memory_tape32"
    supports_cached_feedback = True

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        memory_window: int = 32,
        initialization_seed: int = 4242,
    ):
        super().__init__(backbone)
        if memory_window <= 0:
            raise ValueError("memory_window must be positive")
        self.memory_window = int(memory_window)
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
        yield from self.memory_readers.parameters()

    @staticmethod
    def _cache_next_position(
        past_key_values: tuple[LayerKVCache, ...],
    ) -> int:
        if not past_key_values:
            raise ValueError("past_key_values must not be empty")
        positions = {cache.next_position for cache in past_key_values}
        if len(positions) != 1:
            raise ValueError("layer caches disagree on next absolute position")
        return next(iter(positions))

    def _run_feedback_core(
        self,
        token_embeddings: torch.Tensor,
        memory_states: torch.Tensor,
        *,
        past_key_values: tuple[LayerKVCache, ...] | None,
        use_cache: bool,
        memory_is_bank: bool,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...] | None]:
        if token_embeddings.ndim != 3 or memory_states.ndim != 3:
            raise ValueError("token embeddings and previous memory must be [B,T,D]")
        if token_embeddings.shape[0] != memory_states.shape[0]:
            raise ValueError("token embeddings and previous memory batch sizes differ")
        if token_embeddings.shape[-1] != memory_states.shape[-1]:
            raise ValueError("token embeddings and previous memory dimensions differ")
        if not memory_is_bank and token_embeddings.shape != memory_states.shape:
            raise ValueError("full feedback memory must share token [B,T,D] shape")
        if memory_is_bank and memory_states.shape[1] > self.memory_window:
            raise ValueError("cached memory bank exceeds configured window")
        if past_key_values is not None and len(past_key_values) != len(
            self.backbone.model.layers
        ):
            raise ValueError("past_key_values must contain one cache per layer")

        bsz, seq_len, _ = token_embeddings.shape
        start = (
            0
            if past_key_values is None
            else self._cache_next_position(past_key_values)
        )
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
            past = (
                None
                if past_key_values is None
                else past_key_values[layer_index]
            )
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
                    raise RuntimeError("cached MemoryTape32 layer did not return KV state")
                new_caches.append(cache)

            if memory_is_bank:
                memory_delta = memory_reader.forward_bank(
                    hidden_states, memory_states
                )
            else:
                memory_delta = memory_reader(hidden_states, memory_states)
            hidden_states = hidden_states + memory_delta

            residual = hidden_states
            x = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(x)

        hidden_states = self.backbone.model.norm(hidden_states)
        return (
            hidden_states,
            tuple(new_caches) if new_caches is not None else None,
        )

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids
        hidden, _ = self._run_feedback_core(
            token_embeddings,
            previous_hidden,
            past_key_values=None,
            use_cache=False,
            memory_is_bank=False,
        )
        return hidden

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        del input_ids
        hidden, cache = self._run_feedback_core(
            token_embeddings,
            previous_hidden,
            past_key_values=None,
            use_cache=True,
            memory_is_bank=False,
        )
        if cache is None:
            raise RuntimeError("cached MemoryTape32 prefill did not return KV state")
        return hidden, cache

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        if token_embedding.ndim != 3 or token_embedding.shape[1] != 1:
            raise ValueError("token_embedding must be [B,1,D]")
        hidden, cache = self._run_feedback_core(
            token_embedding,
            feedback_memory,
            past_key_values=past_key_values,
            use_cache=True,
            memory_is_bank=True,
        )
        if cache is None:
            raise RuntimeError("cached MemoryTape32 token did not return KV state")
        return hidden, cache

    def _feedback_memory_from_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids
        if hidden_states.ndim != 3 or hidden_states.shape[1] < 1:
            raise ValueError("hidden_states must be non-empty [B,T,D]")
        keep = min(self.memory_window, hidden_states.shape[1])
        return hidden_states[:, -keep:, :].detach()

    def _append_feedback_memory(
        self,
        feedback_memory: torch.Tensor,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> torch.Tensor:
        del token, position
        if feedback_memory.ndim != 3 or new_hidden.ndim != 3:
            raise ValueError("feedback memory and new hidden must be [B,T,D]")
        if new_hidden.shape[1] != 1:
            raise ValueError("new_hidden must contain exactly one token")
        if (
            feedback_memory.shape[0] != new_hidden.shape[0]
            or feedback_memory.shape[-1] != new_hidden.shape[-1]
        ):
            raise ValueError("feedback memory and new hidden shapes are incompatible")
        combined = torch.cat((feedback_memory, new_hidden), dim=1)
        return combined[:, -self.memory_window :, :].detach()
