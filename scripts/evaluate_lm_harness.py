#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.evaluation.lm_eval_adapter import make_lm_eval_adapter
from tiny_mistral_mptt.model_factory import load_variant


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a checked-in lm-evaluation-harness suite."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", default="evaluation/suites/quick.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    try:
        import lm_eval
    except ImportError as exc:
        raise SystemExit(
            "install evaluation dependencies with: uv sync --extra eval"
        ) from exc

    cfg = load_experiment_config(args.config)
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
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
    adapter = make_lm_eval_adapter(
        model,
        tokenizer_path=Path(cfg.model_dir) / "tokenizer.json",
        device=device,
    )
    suite = yaml.safe_load(Path(args.suite).read_text(encoding="utf-8"))
    collected = {}
    for task in suite["tasks"]:
        name = task["name"]
        result = lm_eval.simple_evaluate(
            model=adapter,
            tasks=[name],
            num_fewshot=task.get("num_fewshot"),
            limit=args.limit,
            log_samples=False,
        )
        collected[name] = result["results"][name]
        print(name, json.dumps(collected[name], sort_keys=True, default=str))
    output = {"suite": str(args.suite), "results": collected}
    text = json.dumps(output, indent=2, sort_keys=True, default=str) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
