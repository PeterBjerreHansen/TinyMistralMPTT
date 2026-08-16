from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from tiny_mistral.modeling import MistralForCausalLM, MistralRMSNorm

from .multipass import MultiPassVariant, shift_previous_hidden


class MemoryAddVariant(MultiPassVariant):
    """Single-state additive previous-pass feedback for pretrained TinyMistral.

    Pass 1 is exact vanilla TinyMistral. On every later pass, position ``t``
    receives a learned residual derived only from the previous pass's top-layer
    state at ``t-1``::

        x_t = e_t + W_M RMSNorm(h^{k-1}_{t-1})

    Position zero has no predecessor and therefore receives an exact zero
    residual. ``memory_projection`` is zero-initialized so all pass depths are
    exact vanilla at initialization. This is intentionally the current repo's
    one-state MemoryAdd control: it reuses the previous top hidden state
    directly rather than adding a separate learned memory-write head.
    """

    variant_name = "memory_add"

    def __init__(self, backbone: MistralForCausalLM):
        super().__init__(backbone)
        hidden_size = int(backbone.config.hidden_size)
        self.memory_norm = MistralRMSNorm(
            hidden_size, eps=float(backbone.config.rms_norm_eps)
        )
        # nn.Linear performs a random default initialization in its constructor.
        # Fork the RNG even though the final projection is zero-initialized so
        # adding this variant never perturbs experiment/data RNG state.
        with torch.random.fork_rng(devices=[]):
            self.memory_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.memory_projection.weight)

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.memory_norm.parameters()
        yield from self.memory_projection.parameters()

    def memory_residual(self, previous_hidden: torch.Tensor) -> torch.Tensor:
        shifted = shift_previous_hidden(previous_hidden)
        return self.memory_projection(self.memory_norm(shifted))

    def feedback_inputs(
        self,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if token_embeddings.shape != previous_hidden.shape:
            raise ValueError(
                "token_embeddings and previous_hidden must have identical [B,T,D] shape"
            )
        return token_embeddings + self.memory_residual(previous_hidden)

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids
        feedback = self.feedback_inputs(token_embeddings, previous_hidden)
        return self.backbone.model(
            inputs_embeds=feedback, use_cache=False
        ).last_hidden_state
