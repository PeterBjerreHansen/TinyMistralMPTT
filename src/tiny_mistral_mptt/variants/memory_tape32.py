from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from tiny_mistral.modeling import MistralForCausalLM, MistralRMSNorm

from ..attention.memory_local import strict_past_local_attention
from .multipass import MultiPassVariant


class MemoryTapeReader(nn.Module):
    """Mistral-shaped GQA cross-attention into a strict-past local memory tape."""

    def __init__(self, backbone: MistralForCausalLM, *, window: int, initialization_seed: int):
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
            self.query_norm = MistralRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.memory_norm = MistralRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
            std = float(config.initializer_range)
            for module in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, hidden_states: torch.Tensor, memory_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape != memory_states.shape:
            raise ValueError("hidden_states and memory_states must share [B,T,D]")
        bsz, seq_len, _ = hidden_states.shape
        query = self.q_proj(self.query_norm(hidden_states))
        memory = self.memory_norm(memory_states)
        key = self.k_proj(memory)
        value = self.v_proj(memory)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
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


class MemoryTape32Variant(MultiPassVariant):
    """Per-layer cross-attention to the previous pass's last 32 deep states."""

    variant_name = "memory_tape32"

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

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids
        if token_embeddings.shape != previous_hidden.shape:
            raise ValueError("token embeddings and previous memory must share [B,T,D]")
        bsz, seq_len, _ = token_embeddings.shape
        position_ids = torch.arange(seq_len, device=token_embeddings.device, dtype=torch.long)
        position_ids = position_ids[None, :].expand(bsz, -1)

        hidden_states = token_embeddings
        for layer, memory_reader in zip(self.backbone.model.layers, self.memory_readers, strict=True):
            residual = hidden_states
            x = layer.input_layernorm(hidden_states)
            x, _ = layer.self_attn(
                x,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                fast_attention_compatible=True,
            )
            hidden_states = residual + x

            hidden_states = hidden_states + memory_reader(hidden_states, previous_hidden)

            residual = hidden_states
            x = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(x)

        return self.backbone.model.norm(hidden_states)
