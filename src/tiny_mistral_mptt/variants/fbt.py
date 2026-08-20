from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from tiny_mistral.modeling import MistralForCausalLM

from .multipass import MultiPassVariant, shift_previous_hidden


class FBTVariant(MultiPassVariant):
    """Full-bandwidth-style asymmetric GLU latent feedback.

    Pass 1 is exact vanilla TinyMistral. On later passes, position ``t`` receives
    the previous pass's top-layer state from ``t-1`` on the value pathway while
    the current token embedding controls the sigmoid gate. Position zero has no
    previous-token state and therefore retains its vanilla token embedding.
    """

    variant_name = "fbt"

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        initialization_seed: int = 4242,
        prefix_mixin_probability: float = 0.0,
    ):
        super().__init__(backbone)
        if not 0.0 <= float(prefix_mixin_probability) <= 1.0:
            raise ValueError("prefix_mixin_probability must be in [0, 1]")
        self.prefix_mixin_probability = float(prefix_mixin_probability)
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

    def feedback_inputs(
        self,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if token_embeddings.shape != previous_hidden.shape:
            raise ValueError(
                "token_embeddings and previous_hidden must have identical [B,T,D] shape"
            )
        shifted = shift_previous_hidden(previous_hidden)
        fused = self.feedback_value(shifted) * torch.sigmoid(
            self.feedback_gate(token_embeddings)
        )
        if fused.shape[1] == 1:
            return token_embeddings
        # Position zero has no previous-token feedback state. Concatenation
        # avoids an in-place overwrite on an autograd-tracked tensor.
        feedback = torch.cat((token_embeddings[:, :1, :], fused[:, 1:, :]), dim=1)
        return self._apply_prefix_mixin(token_embeddings, feedback)

    def _apply_prefix_mixin(
        self,
        token_embeddings: torch.Tensor,
        feedback_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Optionally keep a sampled prefix on the plain embedding path.

        Prefix mixing is an FBT retrofit experiment, not a generic multipass
        operation. Random draws use the CPU generator because experiment
        checkpoints capture that state exactly on every supported device.
        """
        if token_embeddings.shape != feedback_inputs.shape:
            raise ValueError(
                "token_embeddings and feedback_inputs must have identical shapes"
            )
        probability = self.prefix_mixin_probability
        if probability <= 0.0 or feedback_inputs.shape[1] <= 1:
            return feedback_inputs
        should_mix = (
            probability >= 1.0
            or float(torch.rand((), device="cpu")) < probability
        )
        if not should_mix:
            return feedback_inputs
        prefix_length = int(
            torch.randint(1, feedback_inputs.shape[1] + 1, (), device="cpu").item()
        )
        return torch.cat(
            (
                token_embeddings[:, :prefix_length, :],
                feedback_inputs[:, prefix_length:, :],
            ),
            dim=1,
        )

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
