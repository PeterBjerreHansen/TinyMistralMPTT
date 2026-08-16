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
from tiny_mistral_mptt.variants.memory_tape32 import MemoryTape32Variant


def _nll(logits: torch.Tensor, ids: torch.Tensor) -> tuple[float, int]:
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="sum",
    )
    count = int(ids[:, 1:].numel())
    return float(loss.detach().cpu()), count


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal memory interventions for MemoryTape32.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-blocks", type=int, default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if cfg.variant != "memory_tape32":
        raise SystemExit("eval_memory_interventions requires variant=memory_tape32")
    device = resolve_device(cfg.device)
    model = load_variant(
        cfg.variant,
        cfg.model_dir,
        device=device,
        dtype=cfg.dtype,
        attention_backend=cfg.attention_backend,
        architecture_seed=cfg.architecture_seed,
        memory_window=cfg.memory_window,
    )
    if not isinstance(model, MemoryTape32Variant):
        raise SystemExit("eval_memory_interventions requires a MemoryTape32 model")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected = file_sha256(f"{cfg.data_dir}/manifest.json")
    if payload.get("data_manifest_sha256") != expected:
        raise RuntimeError("checkpoint was trained against a different data manifest")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    dataset = PackedTokenDataset(cfg.data_dir, "validation")
    blocks = len(dataset) if args.max_blocks is None else min(len(dataset), args.max_blocks)
    if blocks <= 0:
        raise SystemExit("no validation blocks selected")
    totals = {name: {"loss": 0.0, "count": 0, "delta_sq": 0.0} for name in ("real_memory", "zero_memory", "mismatched_memory")}
    with torch.no_grad():
        for index in range(blocks):
            ids = dataset.batch([index], device=device)
            mismatch_ids = dataset.batch([(index + 1) % len(dataset)], device=device)
            token_embeddings = model.backbone.model.embed_tokens(ids)
            first_hidden = model._run_first_hidden(ids)
            mismatch_hidden = model._run_first_hidden(mismatch_ids)
            real_hidden = model._run_feedback_hidden(ids, token_embeddings, first_hidden)
            zero_hidden = model._run_feedback_hidden(ids, token_embeddings, torch.zeros_like(first_hidden))
            mismatched_hidden = model._run_feedback_hidden(ids, token_embeddings, mismatch_hidden)
            for name, hidden in (
                ("real_memory", real_hidden),
                ("zero_memory", zero_hidden),
                ("mismatched_memory", mismatched_hidden),
            ):
                logits = model.backbone.lm_head(hidden).float()
                loss, count = _nll(logits, ids)
                delta_sq = float(
                    (hidden.float() - first_hidden.float()).square().mean().detach().cpu()
                )
                totals[name]["loss"] += loss
                totals[name]["count"] += count
                totals[name]["delta_sq"] += delta_sq

    result = {}
    for name, values in totals.items():
        nll = values["loss"] / values["count"]
        result[name] = {
            "nll": nll,
            "perplexity": math.exp(nll),
            "hidden_delta_rms": math.sqrt(values["delta_sq"] / blocks),
        }
    result["blocks"] = blocks
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
