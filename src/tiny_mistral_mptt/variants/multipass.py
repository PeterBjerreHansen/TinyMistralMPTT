from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM

from ..training.loss import causal_lm_loss, normalize_pass_weights
from .base import ExperimentalVariant, TrainOutput


@dataclass(frozen=True)
class PassResult:
    hidden_states: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class MultiPassResult:
    passes: tuple[PassResult, ...]

    @property
    def final(self) -> PassResult:
        return self.passes[-1]


def shift_previous_hidden(previous_hidden: torch.Tensor) -> torch.Tensor:
    """Right-shift a [B,T,D] previous-pass state by exactly one token.

    Position zero has no causal predecessor and is therefore filled with zeros.
    This helper defines the shared alignment contract for single-state feedback
    variants such as FBT and MemoryAdd.
    """
    if previous_hidden.ndim != 3:
        raise ValueError("previous_hidden must be [B,T,D]")
    shifted = torch.zeros_like(previous_hidden)
    if previous_hidden.shape[1] > 1:
        shifted[:, 1:, :] = previous_hidden[:, :-1, :]
    return shifted


class MultiPassVariant(ExperimentalVariant):
    """Shared pass recurrence and objective plumbing for research variants.

    Architectures only define how pass ``k>1`` consumes the previous pass's
    final top-layer states. The one-pass path is always the validated vanilla
    TinyMistral backbone.
    """

    supports_cached_feedback = False

    def __init__(self, backbone: MistralForCausalLM):
        super().__init__()
        self.backbone = backbone

    @property
    def config(self):
        return self.backbone.config

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def _run_first_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.model(
            input_ids=input_ids, use_cache=False
        ).last_hidden_state

    def _run_first_hidden_cached(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        output = self.backbone.model(input_ids=input_ids, use_cache=True)
        if output.past_key_values is None:
            raise RuntimeError("cached first pass did not return KV state")
        return output.last_hidden_state, output.past_key_values

    def _run_first_token_cached(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        output = self.backbone.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if output.past_key_values is None:
            raise RuntimeError("cached first-pass token did not return KV state")
        return output.last_hidden_state, output.past_key_values

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        """Run a complete feedback pass while retaining its self-attention cache."""
        raise NotImplementedError

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        """Process one token from an already-strict-past feedback memory."""
        raise NotImplementedError

    def _feedback_memory_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compress a stream history to the state needed by the next pass/token."""
        raise NotImplementedError

    def _append_feedback_memory(
        self, feedback_memory: torch.Tensor, new_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Append one newly produced stream state to its retained feedback memory."""
        raise NotImplementedError

    def _run_hidden_passes(
        self,
        input_ids: torch.Tensor,
        *,
        passes: int,
        phase: str,
    ) -> tuple[torch.Tensor, ...]:
        if input_ids.ndim != 2 or input_ids.shape[1] < 2:
            raise ValueError("input_ids must be [B,T] with at least two tokens")
        if passes < 1:
            raise ValueError("passes must be positive")
        if phase not in {"A", "B"}:
            raise ValueError("phase must be 'A' or 'B'")
        if phase == "A" and passes < 2:
            raise ValueError("Phase A requires at least two passes")

        # In Phase A no added parameter participates in pass 1. The backbone is
        # frozen, so retaining its autograd graph would only waste memory.
        if phase == "A":
            with torch.no_grad():
                first_hidden = self._run_first_hidden(input_ids)
        else:
            first_hidden = self._run_first_hidden(input_ids)

        hidden_states = [first_hidden]
        if passes == 1:
            return tuple(hidden_states)

        token_embeddings = self.backbone.model.embed_tokens(input_ids)
        previous = first_hidden
        for _ in range(1, passes):
            previous = self._run_feedback_hidden(input_ids, token_embeddings, previous)
            hidden_states.append(previous)
        return tuple(hidden_states)

    def compute_passes(
        self,
        input_ids: torch.Tensor,
        *,
        passes: int,
        phase: str = "B",
    ) -> MultiPassResult:
        hidden_states = self._run_hidden_passes(input_ids, passes=passes, phase=phase)
        return MultiPassResult(
            tuple(
                PassResult(
                    hidden_states=hidden, logits=self.backbone.lm_head(hidden).float()
                )
                for hidden in hidden_states
            )
        )

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        *,
        phase: str = "B",
        passes: int = 1,
        loss_weights: Sequence[float] | None = None,
    ) -> TrainOutput:
        hidden_states = self._run_hidden_passes(input_ids, passes=passes, phase=phase)
        pass_losses: list[torch.Tensor] = []
        for hidden in hidden_states:
            logits = self.backbone.lm_head(hidden).float()
            pass_losses.append(causal_lm_loss(logits, input_ids))

        weights = normalize_pass_weights(
            loss_weights,
            passes,
            device=pass_losses[-1].device,
            dtype=pass_losses[-1].dtype,
        )
        loss = sum(
            weight * pass_loss
            for weight, pass_loss in zip(weights, pass_losses, strict=True)
        )
        metrics = {
            f"pass_{index + 1}_loss": float(pass_loss.detach().cpu())
            for index, pass_loss in enumerate(pass_losses)
        }
        metrics.update(
            {
                f"pass_{index + 1}_weight": float(weight.detach().cpu())
                for index, weight in enumerate(weights)
            }
        )
        return TrainOutput(
            loss=loss,
            pass_losses=tuple(pass_losses),
            effective_passes=passes,
            metrics=metrics,
        )

    # Default public model semantics remain the exact one-pass vanilla model.
    # Multipass evaluation is explicit through compute_passes/pass-depth tools.
    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)
