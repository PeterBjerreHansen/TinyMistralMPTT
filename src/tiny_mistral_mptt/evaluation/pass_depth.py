from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

from ..data.packed_dataset import PackedTokenDataset
from ..variants.multipass import MultiPassVariant


@dataclass(frozen=True)
class PassDepthResult:
    passes: int
    blocks: int
    predicted_tokens: int
    nll_by_pass: tuple[float, ...]
    perplexity_by_pass: tuple[float, ...]
    hidden_delta_rms: tuple[float, ...]
    nll_by_source_by_pass: tuple[dict[str, float], ...]

    @property
    def final_nll(self) -> float:
        return self.nll_by_pass[-1]

    @property
    def final_perplexity(self) -> float:
        return self.perplexity_by_pass[-1]


@torch.no_grad()
def evaluate_pass_depth(
    model: MultiPassVariant,
    dataset: PackedTokenDataset,
    *,
    device: torch.device | str,
    passes: int,
    max_blocks: int | None = None,
) -> PassDepthResult:
    if passes < 1:
        raise ValueError("passes must be positive")
    if len(dataset) == 0:
        raise ValueError("validation dataset is empty")
    limit = len(dataset) if max_blocks is None else min(len(dataset), int(max_blocks))
    if limit <= 0:
        raise ValueError("max_blocks leaves no validation blocks")

    was_training = model.training
    model.eval()
    nll_sums = [0.0] * passes
    token_count = 0
    delta_sq_sums = [0.0] * max(passes - 1, 0)
    delta_counts = [0] * max(passes - 1, 0)
    source_nll = [defaultdict(float) for _ in range(passes)]
    source_tokens: dict[int, int] = defaultdict(int)
    try:
        for index in range(limit):
            ids = dataset.batch([index], device=device)
            outputs = model.compute_passes(ids, passes=passes, phase="B")
            labels = model.build_lm_labels(ids)
            count = int(labels.ne(-100).sum().item())
            source_id = dataset.source_id(index)
            token_count += count
            source_tokens[source_id] += count
            for pass_index, pass_output in enumerate(outputs.passes):
                logits = pass_output.logits.float()
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.to(logits.device).reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                value = float(loss_sum.detach().cpu())
                nll_sums[pass_index] += value
                source_nll[pass_index][source_id] += value
            for transition in range(1, passes):
                delta = outputs.passes[transition].hidden_states - outputs.passes[transition - 1].hidden_states
                delta_sq_sums[transition - 1] += float(delta.float().pow(2).sum().cpu())
                delta_counts[transition - 1] += delta.numel()
    finally:
        model.train(was_training)

    nll = tuple(value / token_count for value in nll_sums)
    id_to_name = {value: key for key, value in dataset.manifest.source_ids.items()}
    by_source: list[dict[str, float]] = []
    for pass_index in range(passes):
        by_source.append(
            {
                id_to_name[source_id]: source_nll[pass_index][source_id] / source_tokens[source_id]
                for source_id in sorted(source_tokens)
            }
        )
    return PassDepthResult(
        passes=passes,
        blocks=limit,
        predicted_tokens=token_count,
        nll_by_pass=nll,
        perplexity_by_pass=tuple(math.exp(min(value, 50.0)) for value in nll),
        hidden_delta_rms=tuple(
            math.sqrt(total / count) for total, count in zip(delta_sq_sums, delta_counts, strict=True)
        ),
        nll_by_source_by_pass=tuple(by_source),
    )
