#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from tiny_mistral.loading import verify_target_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the exact pinned TinyMistral checkpoint target.")
    parser.add_argument("model_dir", nargs="?", default="checkpoints/TinyMistral-248M-v3")
    args = parser.parse_args()
    result = verify_target_checkpoint(args.model_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
