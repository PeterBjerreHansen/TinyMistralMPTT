#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math

import torch

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.manifest import file_sha256
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.model_factory import load_variant
from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant
from tiny_mistral_mptt.variants.memory_tape32 import MemoryTape32Variant


def _nll(logits: torch.Tensor, ids: torch.Tensor) -> tuple[float, int]:
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="sum",
    )
    count = int(ids[:, 1:].numel())
    return float(loss.detach().cpu()), count


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run real/zero/mismatched previous-state interventions for "
            "MemoryAdd or MemoryTape32."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-blocks", type=int, default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if cfg.variant not in {"memory_add", "memory_tape32"}:
        raise SystemExit(
            "evaluate_memory_interventions requires variant=memory_add or memory_tape32"
        )
    device = resolve_device(cfg.device)
    model = load_variant(
        cfg.variant,
        cfg.model_dir,
        device=device,
        dtype=cfg.dtype,
        attention_backend=cfg.attention_backend,
        architecture_seed=cfg.architecture_seed,
        memory_window=cfg.memory_window,
        prefix_mixin_probability=cfg.prefix_mixin_probability,
    )
    if not isinstance(model, (MemoryAddVariant, MemoryTape32Variant)):
        raise SystemExit("loaded model does not support memory interventions")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected = file_sha256(f"{cfg.data_dir}/manifest.json")
    if payload.get("data_manifest_sha256") != expected:
        raise RuntimeError("checkpoint was trained against a different data manifest")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    dataset = PackedTokenDataset(cfg.data_dir, "validation")
    blocks = (
        len(dataset) if args.max_blocks is None else min(len(dataset), args.max_blocks)
    )
    if blocks <= 0:
        raise SystemExit("no validation blocks selected")

    names = ("real_memory", "zero_memory", "mismatched_memory")
    totals = {
        name: {"loss": 0.0, "count": 0, "delta_sq": 0.0}
        for name in names
    }
    baseline_loss = 0.0
    baseline_count = 0
    embedding_rms_sum = 0.0
    residual_rms_sum = 0.0

    with torch.no_grad():
        for index in range(blocks):
            ids = dataset.batch([index], device=device)
            mismatch_ids = dataset.batch([(index + 1) % len(dataset)], device=device)
            token_embeddings = model.backbone.model.embed_tokens(ids)
            first_hidden = model._run_first_hidden(ids)
            mismatch_hidden = model._run_first_hidden(mismatch_ids)

            baseline_logits = model.backbone.lm_head(first_hidden).float()
            loss, count = _nll(baseline_logits, ids)
            baseline_loss += loss
            baseline_count += count

            real_hidden = model._run_feedback_hidden(
                ids, token_embeddings, first_hidden
            )
            zero_hidden = model._run_feedback_hidden(
                ids, token_embeddings, torch.zeros_like(first_hidden)
            )
            mismatched_hidden = model._run_feedback_hidden(
                ids, token_embeddings, mismatch_hidden
            )
            for name, hidden in (
                ("real_memory", real_hidden),
                ("zero_memory", zero_hidden),
                ("mismatched_memory", mismatched_hidden),
            ):
                logits = model.backbone.lm_head(hidden).float()
                loss, count = _nll(logits, ids)
                delta_sq = float(
                    (hidden.float() - first_hidden.float())
                    .square()
                    .mean()
                    .detach()
                    .cpu()
                )
                totals[name]["loss"] += loss
                totals[name]["count"] += count
                totals[name]["delta_sq"] += delta_sq

            if isinstance(model, MemoryAddVariant):
                embedding_rms_sum += _rms(token_embeddings[:, 1:, :])
                residual_rms_sum += _rms(
                    model.memory_residual(first_hidden)[:, 1:, :]
                )

    result: dict[str, object] = {
        "variant": cfg.variant,
        "blocks": blocks,
        "baseline_pass1": {
            "nll": baseline_loss / baseline_count,
            "perplexity": math.exp(baseline_loss / baseline_count),
        },
    }
    for name, values in totals.items():
        nll = values["loss"] / values["count"]
        result[name] = {
            "nll": nll,
            "perplexity": math.exp(nll),
            "hidden_delta_rms": math.sqrt(values["delta_sq"] / blocks),
        }

    if isinstance(model, MemoryAddVariant):
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
