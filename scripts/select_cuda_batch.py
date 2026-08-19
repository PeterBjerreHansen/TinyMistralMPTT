#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_mistral_mptt.efficiency import recommend_cuda_microbatch


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the smallest common efficient CUDA microbatch from a batch-qualification result."
        )
    )
    parser.add_argument("result", help="JSON emitted by benchmark_training_efficiency.py")
    parser.add_argument("--efficiency-fraction", type=float, default=0.90)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--reference-optimizer-batch-tokens", type=int, default=2048)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    document = json.loads(Path(args.result).read_text(encoding="utf-8"))
    recommendation = recommend_cuda_microbatch(
        document,
        passes=args.passes,
        sequence_length=args.sequence_length,
        efficiency_fraction=args.efficiency_fraction,
        reference_optimizer_batch_tokens=args.reference_optimizer_batch_tokens,
    )
    payload = recommendation.to_dict()
    if recommendation.changes_optimizer_batch:
        payload["protocol_action"] = (
            "qualify the proposed optimizer-batch size before locking a core run"
        )
    else:
        payload["protocol_action"] = "optimizer-batch size matches the validated reference"

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
