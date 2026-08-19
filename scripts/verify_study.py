#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tiny_mistral_mptt.studies import discover_studies, verify_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate colocated benchmark STUDY.yaml manifests and runnable arms."
    )
    parser.add_argument(
        "studies",
        nargs="*",
        help="study directory or STUDY.yaml path; defaults to all development/core studies",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = [Path(value) for value in args.studies] or discover_studies(root)
    if not paths:
        raise SystemExit("no STUDY.yaml manifests found")
    for path in paths:
        result = verify_study(path)
        rel = result.manifest_path.resolve().relative_to(root)
        print(
            f"PASS: {rel} name={result.name} status={result.status} "
            f"arms={len(result.arm_ids)} comparisons={len(result.comparison_names)}"
        )


if __name__ == "__main__":
    main()
