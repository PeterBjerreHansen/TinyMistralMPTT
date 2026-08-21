from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_mistral.modeling import MistralRMSNorm


NMP_TARGET_NORMALIZATIONS = {"none", "rms"}


def normalize_nmp_target(
    states: torch.Tensor,
    *,
    normalization: str,
    eps: float,
) -> torch.Tensor:
    """Apply the configured parameter-free normalization to NMP targets.

    ``rms`` deliberately matches the variance calculation used by the
    model's RMSNorm, but omits its learned feature-wise gain.  Target
    normalization must not add trainable parameters or create a gradient path
    through the future memory.  ``none`` is the exact stored representation.
    """

    if normalization not in NMP_TARGET_NORMALIZATIONS:
        raise ValueError(
            "NMP target normalization must be one of "
            f"{sorted(NMP_TARGET_NORMALIZATIONS)}"
        )
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("NMP target normalization eps must be finite and positive")
    if not states.is_floating_point():
        raise ValueError("NMP targets must have a floating-point dtype")
    if normalization == "none":
        return states
    values = states.to(torch.float32)
    variance = values.square().mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + float(eps))
    return normalized.to(states.dtype)


class LatentPredictionHead(nn.Module):
    """Predict a future memory from one current top-layer hidden state.

    The deliberately narrow ``forward(hidden_states)`` API is part of the
    causality contract: token embeddings and future observations cannot enter
    this prediction path.  The final projection starts at zero so enabling NMP
    does not inject an arbitrary latent prediction into the first update.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        projection_factor: float,
        rms_norm_eps: float,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if not math.isfinite(float(projection_factor)) or projection_factor <= 0:
            raise ValueError("projection_factor must be finite and positive")
        width = max(128, int(math.ceil(hidden_size * projection_factor / 128.0)) * 128)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.input_norm = MistralRMSNorm(hidden_size, eps=float(rms_norm_eps))
            self.hidden_1 = nn.Linear(hidden_size, width)
            self.hidden_2 = nn.Linear(width, width)
            self.output = nn.Linear(width, hidden_size)
            nn.init.xavier_uniform_(self.hidden_1.weight)
            nn.init.zeros_(self.hidden_1.bias)
            nn.init.xavier_uniform_(self.hidden_2.weight)
            nn.init.zeros_(self.hidden_2.bias)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("NMP predictor input must be [B,T,D]")
        hidden = F.gelu(self.hidden_1(self.input_norm(hidden_states)))
        hidden = F.gelu(self.hidden_2(hidden))
        return self.output(hidden)


def next_strict_true_indices(mask: torch.Tensor) -> torch.Tensor:
    """Index of the first true position strictly to the right, or ``T``.

    This device-side suffix scan is shared by recurrent and tape NMP.  The
    one-position shift is essential: a write at query position ``t`` can never
    become that query's target.
    """

    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [B,T]")
    batch, length = mask.shape
    positions = torch.arange(length, device=mask.device, dtype=torch.long)
    sentinel = torch.full((batch, length), length, device=mask.device, dtype=torch.long)
    candidates = torch.where(mask, positions[None, :].expand(batch, -1), sentinel)
    suffix_min = torch.flip(
        torch.cummin(torch.flip(candidates, dims=(1,)), dim=1).values,
        dims=(1,),
    )
    return torch.cat(
        (
            suffix_min[:, 1:],
            torch.full((batch, 1), length, device=mask.device, dtype=torch.long),
        ),
        dim=1,
    )


def _gather_states(states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    safe = indices.clamp(max=states.shape[1] - 1)
    return states.gather(1, safe[:, :, None].expand(-1, -1, states.shape[-1]))


def target_diagnostics(targets: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return RMS magnitude and mean feature standard deviation of valid targets."""

    selected = targets[valid]
    if selected.shape[0] == 0:
        raise ValueError("NMP batch has no valid future-memory targets")
    rms = selected.float().square().mean().sqrt()
    # unbiased=False remains defined for a single target vector.
    feature_std = selected.float().std(dim=0, unbiased=False).mean()
    return rms, feature_std


def recurrent_nmp_pass_loss(
    predictions: torch.Tensor,
    *,
    final_targets: torch.Tensor,
    ordinary_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth-L1 loss from each ordinary ``t`` to final memory at ordinary ``t+1``."""

    if predictions.shape != final_targets.shape:
        raise ValueError("recurrent NMP predictions and targets must have equal [B,T,D] shape")
    if ordinary_mask.shape != predictions.shape[:2]:
        raise ValueError("ordinary mask shape differs from recurrent NMP states")
    next_index = next_strict_true_indices(ordinary_mask)
    valid = ordinary_mask & next_index.lt(predictions.shape[1])
    targets = _gather_states(final_targets, next_index).detach()
    if not bool(valid.any()):
        raise ValueError("recurrent NMP requires at least two linguistic tokens per example batch")
    loss = F.smooth_l1_loss(predictions[valid].float(), targets[valid].float())
    rms, feature_std = target_diagnostics(targets, valid)
    return loss, rms, feature_std


def tape_nmp_pass_loss(
    predictions: torch.Tensor,
    *,
    final_written_states: torch.Tensor,
    ordinary_mask: torch.Tensor,
    write_mask: torch.Tensor,
    sequence_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Target-balanced loss to the first strictly-future written memory.

    Query positions are linguistic tokens.  Each future write contributes
    equally within an example even if many queries map to that same write; then
    examples with at least one target contribute equally to the batch loss.
    """

    if predictions.shape != final_written_states.shape:
        raise ValueError("tape NMP predictions and targets must have equal [B,T,D] shape")
    if ordinary_mask.shape != predictions.shape[:2] or write_mask.shape != ordinary_mask.shape:
        raise ValueError("tape NMP masks must match [B,T]")
    if sequence_positions.shape != ordinary_mask.shape:
        raise ValueError("tape NMP sequence positions must match [B,T]")

    batch, length, _ = predictions.shape
    next_write = next_strict_true_indices(write_mask)
    valid = ordinary_mask & next_write.lt(length)
    if not bool(valid.any()):
        raise ValueError("tape NMP batch has no strictly-future write target")
    targets = _gather_states(final_written_states, next_write).detach()
    per_query = F.smooth_l1_loss(
        predictions.float(), targets.float(), reduction="none"
    ).mean(dim=-1)

    safe_write = next_write.clamp(max=length - 1)
    event_sums = per_query.new_zeros((batch, length)).scatter_add(
        1, safe_write, per_query * valid
    )
    event_counts = per_query.new_zeros((batch, length)).scatter_add(
        1, safe_write, valid.to(per_query.dtype)
    )
    present = event_counts.gt(0)
    event_means = event_sums / event_counts.clamp_min(1)
    example_counts = present.sum(dim=1)
    example_means = (event_means * present).sum(dim=1) / example_counts.clamp_min(1)
    has_target = example_counts.gt(0)
    loss = example_means[has_target].mean()

    target_positions = sequence_positions.gather(1, safe_write)
    distances = target_positions - sequence_positions
    if bool((distances[valid] < 0).any()):
        raise RuntimeError("tape NMP produced a negative linguistic distance")
    buckets = {
        "0": distances.eq(0),
        "1": distances.eq(1),
        "2_4": distances.ge(2) & distances.le(4),
        "5_8": distances.ge(5) & distances.le(8),
        "9_16": distances.ge(9) & distances.le(16),
        "17_32": distances.ge(17) & distances.le(32),
        "33_plus": distances.ge(33),
    }
    distance_losses = {
        name: per_query[valid & bucket].mean()
        for name, bucket in buckets.items()
        if bool((valid & bucket).any())
    }
    rms, feature_std = target_diagnostics(targets, valid)
    return loss, rms, feature_std, distance_losses
