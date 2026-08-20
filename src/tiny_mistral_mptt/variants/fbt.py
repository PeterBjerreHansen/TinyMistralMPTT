from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM, MistralRMSNorm

from .multipass import MultiPassVariant, shift_previous_hidden


class FBTVariant(MultiPassVariant):
    """Full-bandwidth-style asymmetric GLU latent feedback.

    Pass 1 is exact vanilla TinyMistral. On later passes, position ``t`` receives
    the previous pass's top-layer state from ``t-1`` on the value pathway while
    the current token embedding controls the sigmoid gate. Position zero has no
    previous-token state and therefore retains its vanilla token embedding.
    """

    variant_name = "fbt"
    supports_cached_feedback = True

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
            self.feedback_input_norm = MistralRMSNorm(
                hidden_size, eps=float(backbone.config.rms_norm_eps)
            )
            std = float(backbone.config.initializer_range)
            nn.init.normal_(self.feedback_value.weight, mean=0.0, std=std)
            nn.init.normal_(self.feedback_gate.weight, mean=0.0, std=std)

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.feedback_value.parameters()
        yield from self.feedback_gate.parameters()
        yield from self.feedback_input_norm.parameters()

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
        fused = self.feedback_input_norm(fused)
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
        if not self.training or probability <= 0.0 or feedback_inputs.shape[1] <= 1:
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

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        del input_ids
        feedback = self.feedback_inputs(token_embeddings, previous_hidden)
        output = self.backbone.model(inputs_embeds=feedback, use_cache=True)
        if output.past_key_values is None:
            raise RuntimeError("cached FBT prefill did not return KV state")
        return output.last_hidden_state, output.past_key_values

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        del token
        if token_embedding.ndim != 3 or token_embedding.shape[1] != 1:
            raise ValueError("token_embedding must be [B,1,D]")
        if feedback_memory.shape != token_embedding.shape:
            raise ValueError("FBT cached feedback memory must be [B,1,D]")
        feedback = self.feedback_value(feedback_memory) * torch.sigmoid(
            self.feedback_gate(token_embedding)
        )
        feedback = self.feedback_input_norm(feedback)
        output = self.backbone.model(
            inputs_embeds=feedback,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if output.past_key_values is None:
            raise RuntimeError("cached FBT token did not return KV state")
        return output.last_hidden_state, output.past_key_values

    def _feedback_memory_from_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids
        if hidden_states.ndim != 3 or hidden_states.shape[1] < 1:
            raise ValueError("hidden_states must be non-empty [B,T,D]")
        return hidden_states[:, -1:, :].detach()

    def _append_feedback_memory(
        self,
        feedback_memory: torch.Tensor,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> torch.Tensor:
        del token, position
        if feedback_memory.ndim != 3 or feedback_memory.shape[1] != 1:
            raise ValueError("FBT feedback memory must be [B,1,D]")
        if new_hidden.ndim != 3 or new_hidden.shape[1] != 1:
            raise ValueError("new_hidden must be [B,1,D]")
        if feedback_memory.shape != new_hidden.shape:
            raise ValueError("feedback memory and new hidden shapes differ")
        return new_hidden.detach()
