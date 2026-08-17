"""Historical FBT calibrated-initialization procedure.

This module is intentionally outside ``src/``. It preserves the one-off
calibration used in the recorded FBT retrofit experiment without expanding the
stable experiment configuration or model API.
"""

from __future__ import annotations

import torch

from tiny_mistral_mptt.variants.fbt import FBTVariant
from tiny_mistral_mptt.variants.multipass import shift_previous_hidden


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().detach().cpu())


def _std(value: torch.Tensor) -> float:
    return float(value.float().std(unbiased=False).detach().cpu())


def calibrate_fbt_initialization(
    model: FBTVariant,
    input_ids: torch.Tensor,
    *,
    gate_logit_std_target: float = 1.0,
) -> dict[str, float]:
    """Rescale only the two added FBT matrices on one fixed batch."""
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("input_ids must be [B,T] with at least two tokens")
    if not torch.isfinite(torch.tensor(gate_logit_std_target)) or gate_logit_std_target <= 0:
        raise ValueError("gate_logit_std_target must be finite and positive")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        token_embeddings = model.backbone.model.embed_tokens(input_ids)
        previous_hidden = model._run_first_hidden(input_ids)
        shifted = shift_previous_hidden(previous_hidden)
        value = model.feedback_value(shifted)
        gate_logits = model.feedback_gate(token_embeddings)

        non_initial = (slice(None), slice(1, None), slice(None))
        embedding_rms = _rms(token_embeddings[non_initial])
        pre_value_rms = _rms(value[non_initial])
        pre_gate_logit_std = _std(gate_logits)
        pre_gate = torch.sigmoid(gate_logits)
        pre_fused_rms = _rms((value * pre_gate)[non_initial])

        if pre_gate_logit_std <= torch.finfo(torch.float32).eps:
            raise RuntimeError("cannot calibrate FBT gate with zero logit variation")
        gate_scale = float(gate_logit_std_target) / pre_gate_logit_std
        model.feedback_gate.weight.mul_(gate_scale)

        gate_logits = model.feedback_gate(token_embeddings)
        gate = torch.sigmoid(gate_logits)
        fused_rms = _rms((value * gate)[non_initial])
        if fused_rms <= torch.finfo(torch.float32).eps:
            raise RuntimeError("cannot calibrate FBT value pathway with zero fused RMS")
        value_scale = embedding_rms / fused_rms
        model.feedback_value.weight.mul_(value_scale)

        value = model.feedback_value(shifted)
        fused = value * gate
        stats = {
            "embedding_rms": embedding_rms,
            "pre_value_rms": pre_value_rms,
            "pre_gate_logit_std": pre_gate_logit_std,
            "pre_gate_std": _std(pre_gate),
            "pre_fused_rms": pre_fused_rms,
            "gate_scale": gate_scale,
            "value_scale": value_scale,
            "post_gate_logit_std": _std(gate_logits),
            "post_gate_std": _std(gate),
            "post_fused_rms": _rms(fused[non_initial]),
        }

    model.train(was_training)
    return stats
