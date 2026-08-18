#!/usr/bin/env python
from __future__ import annotations

import argparse

from tiny_mistral.loading import MODEL_ID, MODEL_REVISION, download_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned TinyMistral checkpoint files.")
    parser.add_argument("--output", default="checkpoints/TinyMistral-248M-v3")
    parser.add_argument("--repo-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()
    path = download_snapshot(args.output, repo_id=args.repo_id, revision=args.revision)
    print(path)


if __name__ == "__main__":
    main()
