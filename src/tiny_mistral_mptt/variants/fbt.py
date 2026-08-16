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

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        initialization_seed: int = 4242,
        prefix_mixin_probability: float = 0.0,
    ):
        super().__init__(backbone, prefix_mixin_probability=prefix_mixin_probability)
        self.initialization_stats: dict[str, float] | None = None
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
        feedback = torch.cat((token_embeddings[:, :1, :], fused[:, 1:, :]), dim=1)
        return self.apply_prefix_mixin(token_embeddings, feedback)

    @staticmethod
    def _rms(value: torch.Tensor) -> float:
        return float(value.float().square().mean().sqrt().detach().cpu())

    @staticmethod
    def _std(value: torch.Tensor) -> float:
        return float(value.float().std(unbiased=False).detach().cpu())

    def calibrate_initialization(
        self,
        input_ids: torch.Tensor,
        *,
        gate_logit_std_target: float = 1.0,
    ) -> dict[str, float]:
        """Rescale only the added FBT matrices using one fixed calibration batch.

        ``feedback_value`` is calibrated after the gate scale so the final
        fused feedback RMS matches the token-embedding RMS on non-initial
        positions. The gate matrix is scaled to a configurable logit standard
        deviation, which avoids a gate that is effectively constant at 0.5
        when the pretrained embedding RMS is unusually small.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] < 2:
            raise ValueError("input_ids must be [B,T] with at least two tokens")
        if not torch.isfinite(torch.tensor(gate_logit_std_target)) or gate_logit_std_target <= 0:
            raise ValueError("gate_logit_std_target must be finite and positive")

        was_training = self.training
        self.eval()
        with torch.no_grad():
            token_embeddings = self.backbone.model.embed_tokens(input_ids)
            previous_hidden = self._run_first_hidden(input_ids)
            shifted = self.shift_previous(previous_hidden)
            value = self.feedback_value(shifted)
            gate_logits = self.feedback_gate(token_embeddings)

            non_initial = (slice(None), slice(1, None), slice(None))
            embedding_rms = self._rms(token_embeddings[non_initial])
            pre_value_rms = self._rms(value[non_initial])
            pre_gate_logit_std = self._std(gate_logits)
            pre_gate = torch.sigmoid(gate_logits)
            pre_fused_rms = self._rms((value * pre_gate)[non_initial])

            if pre_gate_logit_std <= torch.finfo(torch.float32).eps:
                raise RuntimeError("cannot calibrate FBT gate with zero logit variation")
            gate_scale = float(gate_logit_std_target) / pre_gate_logit_std
            self.feedback_gate.weight.mul_(gate_scale)

            gate_logits = self.feedback_gate(token_embeddings)
            gate = torch.sigmoid(gate_logits)
            fused_before_value_rescale = value * gate
            fused_rms_before_value_rescale = self._rms(fused_before_value_rescale[non_initial])
            if fused_rms_before_value_rescale <= torch.finfo(torch.float32).eps:
                raise RuntimeError("cannot calibrate FBT value pathway with zero fused RMS")
            value_scale = embedding_rms / fused_rms_before_value_rescale
            self.feedback_value.weight.mul_(value_scale)

            value = self.feedback_value(shifted)
            fused = value * gate
            post_gate_logit_std = self._std(gate_logits)
            post_gate = torch.sigmoid(gate_logits)
            post_fused_rms = self._rms(fused[non_initial])
            post_gate_std = self._std(post_gate)

        if was_training:
            self.train()
        return {
            "embedding_rms": embedding_rms,
            "pre_value_rms": pre_value_rms,
            "pre_gate_logit_std": pre_gate_logit_std,
            "pre_gate_std": self._std(pre_gate),
            "pre_fused_rms": pre_fused_rms,
            "gate_scale": gate_scale,
            "value_scale": value_scale,
            "post_gate_logit_std": post_gate_logit_std,
            "post_gate_std": post_gate_std,
            "post_fused_rms": post_fused_rms,
        }

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids
        feedback = self.feedback_inputs(token_embeddings, previous_hidden)
        return self.backbone.model(inputs_embeds=feedback, use_cache=False).last_hidden_state
