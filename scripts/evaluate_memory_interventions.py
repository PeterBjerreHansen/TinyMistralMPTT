#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math

import torch

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.manifest import file_sha256
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant
from tiny_mistral_mptt.variants.tape_add_hybrid import TapeAddHybridVariant
from tiny_mistral_mptt.variants.tape import TapeVariant


SUPPORTED = {"memory_add", "tape", "tape_add_hybrid"}


def _nll(model, logits: torch.Tensor, ids: torch.Tensor) -> tuple[float, int]:
    labels = model.build_lm_labels(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.to(logits.device).reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    count = int(labels.ne(-100).sum().item())
    return float(loss.detach().cpu()), count


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().detach().cpu())


def _condition_hiddens(
    model,
    ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    real: torch.Tensor,
    mismatch: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = torch.zeros_like(real)
    if isinstance(model, TapeAddHybridVariant):
        return {
            "real_memory": model._run_feedback_hidden_components(
                ids, token_embeddings, fast_hidden=real, tape_hidden=real
            ),
            "zero_memory": model._run_feedback_hidden_components(
                ids, token_embeddings, fast_hidden=zero, tape_hidden=zero
            ),
            "mismatched_memory": model._run_feedback_hidden_components(
                ids, token_embeddings, fast_hidden=mismatch, tape_hidden=mismatch
            ),
            "zero_fast_real_tape": model._run_feedback_hidden_components(
                ids, token_embeddings, fast_hidden=zero, tape_hidden=real
            ),
            "mismatched_fast_real_tape": model._run_feedback_hidden_components(
                ids, token_embeddings, fast_hidden=mismatch, tape_hidden=real
            ),
            "real_fast_zero_tape": model._run_feedback_hidden_components(
                ids, token_embeddings, fast_hidden=real, tape_hidden=zero
            ),
            "real_fast_mismatched_tape": model._run_feedback_hidden_components(
                ids, token_embeddings, fast_hidden=real, tape_hidden=mismatch
            ),
        }
    return {
        "real_memory": model._run_feedback_hidden(ids, token_embeddings, real),
        "zero_memory": model._run_feedback_hidden(ids, token_embeddings, zero),
        "mismatched_memory": model._run_feedback_hidden(ids, token_embeddings, mismatch),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run real/zero/mismatched recurrence interventions. The hybrid also "
            "reports fast and tape channel interventions independently."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-blocks", type=int, default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if cfg.variant not in SUPPORTED:
        raise SystemExit(
            "evaluate_memory_interventions requires a MemoryAdd/Tape variant"
        )
    device = resolve_device(cfg.device)
    model = load_variant_from_config(cfg, device=device)
    if not isinstance(
        model,
        (MemoryAddVariant, TapeVariant, TapeAddHybridVariant),
    ):
        raise SystemExit("loaded model does not support memory interventions")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected = file_sha256(f"{cfg.data_dir}/manifest.json")
    if payload.get("data_manifest_sha256") != expected:
        raise RuntimeError("checkpoint was trained against a different data manifest")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    dataset = load_packed_dataset_for_experiment(cfg.data_dir, "validation", memory_write_mode=cfg.memory_write_mode, memory_write_stride=cfg.memory_write_stride)
    blocks = len(dataset) if args.max_blocks is None else min(len(dataset), args.max_blocks)
    if blocks <= 0:
        raise SystemExit("no validation blocks selected")

    totals: dict[str, dict[str, float | int]] = {}
    baseline_loss = 0.0
    baseline_count = 0
    embedding_rms_sum = 0.0
    residual_rms_sum = 0.0

    with torch.no_grad():
        for index in range(blocks):
            ids = dataset.batch([index], device=device)
            mismatch_ids = dataset.batch([(index + 1) % len(dataset)], device=device)
            token_embeddings = model.input_embeddings(ids)
            first_hidden = model._run_first_hidden(ids)
            mismatch_hidden = model._run_first_hidden(mismatch_ids)

            baseline_logits = model.backbone.lm_head(first_hidden).float()
            loss, count = _nll(model, baseline_logits, ids)
            baseline_loss += loss
            baseline_count += count

            conditions = _condition_hiddens(
                model, ids, token_embeddings, first_hidden, mismatch_hidden
            )
            for name, hidden in conditions.items():
                values = totals.setdefault(
                    name, {"loss": 0.0, "count": 0, "delta_sq": 0.0}
                )
                logits = model.backbone.lm_head(hidden).float()
                loss, count = _nll(model, logits, ids)
                delta_sq = float(
                    (hidden.float() - first_hidden.float())
                    .square()
                    .mean()
                    .detach()
                    .cpu()
                )
                values["loss"] = float(values["loss"]) + loss
                values["count"] = int(values["count"]) + count
                values["delta_sq"] = float(values["delta_sq"]) + delta_sq

            if isinstance(model, (MemoryAddVariant, TapeAddHybridVariant)):
                embedding_rms_sum += _rms(token_embeddings[:, 1:, :])
                residual_rms_sum += _rms(model.memory_residual(first_hidden, ids)[:, 1:, :] if isinstance(model, TapeAddHybridVariant) else model.memory_residual(first_hidden)[:, 1:, :])

    result: dict[str, object] = {
        "variant": cfg.variant,
        "blocks": blocks,
        "baseline_pass1": {
            "nll": baseline_loss / baseline_count,
            "perplexity": math.exp(baseline_loss / baseline_count),
        },
    }
    for name, values in totals.items():
        count = int(values["count"])
        nll = float(values["loss"]) / count
        result[name] = {
            "nll": nll,
            "perplexity": math.exp(nll),
            "hidden_delta_rms": math.sqrt(float(values["delta_sq"]) / blocks),
        }

    if isinstance(model, (MemoryAddVariant, TapeAddHybridVariant)):
        embedding_rms = embedding_rms_sum / blocks
        residual_rms = residual_rms_sum / blocks
        result["memory_add_scales"] = {
            "embedding_rms_noninitial": embedding_rms,
            "memory_residual_rms_noninitial": residual_rms,
            "residual_to_embedding_rms_ratio": (
                residual_rms / embedding_rms if embedding_rms > 0 else float("nan")
            ),
        }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
