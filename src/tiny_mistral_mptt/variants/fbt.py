from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from tiny_mistral.modeling import MistralForCausalLM

from .multipass import MultiPassVariant


class FBTVariant(MultiPassVariant):
    """Full-bandwidth-style asymmetric GLU latent feedback.

    Pass 1 is exact vanilla TinyMistral. On later passes, position ``t`` receives
    the previous pass's top-layer state from ``t-1`` on the value pathway while
    the current token embedding controls the sigmoid gate. Position zero has no
    previous-token state and therefore retains its vanilla token embedding.
    """

    variant_name = "fbt"

    def __init__(self, backbone: MistralForCausalLM, *, initialization_seed: int = 4242):
        super().__init__(backbone)
        hidden_size = int(backbone.config.hidden_size)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.feedback_value = nn.Linear(hidden_size, hidden_size, bias=False)
            self.feedback_gate = nn.Linear(hidden_size, hidden_size, bias=False)
            std = float(backbone.config.initializer_range)
            nn.init.normal_(self.feedback_value.weight, mean=0.0, std=std)
            nn.init.normal_(self.feedback_gate.weight, mean=0.0, std=std)

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.feedback_value.parameters()
        yield from self.feedback_gate.parameters()

    @staticmethod
    def shift_previous(previous_hidden: torch.Tensor) -> torch.Tensor:
        if previous_hidden.ndim != 3:
            raise ValueError("previous_hidden must be [B,T,D]")
        shifted = torch.zeros_like(previous_hidden)
        if previous_hidden.shape[1] > 1:
            shifted[:, 1:, :] = previous_hidden[:, :-1, :]
        return shifted

    def feedback_inputs(
        self,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if token_embeddings.shape != previous_hidden.shape:
            raise ValueError("token_embeddings and previous_hidden must have identical [B,T,D] shape")
        shifted = self.shift_previous(previous_hidden)
        fused = self.feedback_value(shifted) * torch.sigmoid(self.feedback_gate(token_embeddings))
        if fused.shape[1] == 1:
            return token_embeddings
        # Position zero has no previous-token feedback state. Concatenation
        # avoids an in-place overwrite on an autograd-tracked tensor.
        return torch.cat((token_embeddings[:, :1, :], fused[:, 1:, :]), dim=1)

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids
        feedback = self.feedback_inputs(token_embeddings, previous_hidden)
        return self.backbone.model(inputs_embeds=feedback, use_cache=False).last_hidden_state
