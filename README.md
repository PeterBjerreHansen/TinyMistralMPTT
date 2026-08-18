# TinyMistralMPTT

Research code for MemoryAdd, MemoryTape32, and recurrent inference on a
validated TinyMistral backbone.

## Read first

- `configs/`: canonical runnable configs.
- `experiments/stage2_cleanroom_v1/`: complete protocol lineage, controls, and
  results.
- `runs/`: generated checkpoints and metrics; ignored by Git.
- `docs/`: durable implementation and provenance contracts.
- `src/tiny_mistral/`: validated vendored vanilla implementation.
- `src/tiny_mistral_mptt/`: research models, training, evaluation, and inference.

The checkpoint directory is `checkpoints/TinyMistral-248M-v3/`. This exact name
is retained because it is the upstream model identifier, not this repository's
version. The upstream revision is pinned in `docs/UPSTREAMS.md`.

## Current Stage 2 status

The Stage 2 protocol remains open. E2 selected `1e-6` for both backbone and
added parameters, and the completed 512-token K sweep makes K=2 the provisional
winner. The K schedule is intentionally still pending while context-length and
efficiency qualification are underway; no 100M-token main campaign is locked.

The active evidence is documented in:

```text
experiments/stage2_cleanroom_v1/results/k_sweep.md
experiments/stage2_cleanroom_v1/PROTOCOL.yaml
```

Do not start a main Stage 2 run from `configs/stage2/` until the context
qualification is complete and the protocol is explicitly relocked.

## Validate

```bash
uv sync --extra data --extra eval
uv run pytest -q
uv run python -m compileall -q src scripts tests experiments
git diff --check
```

Prepare and verify the clean-room data artifact:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/verify_data.py data/stage2_cleanroom_v1/sequence_512
```

Run the engineering efficiency battery on the local accelerator with the
targets in `efficiency_benchmarks/`. These measurements are separate from the
scientific experiment record.

The CUDA substrate config and `docs/CLOUD.md` describe the paid-run preflight
path. The cloud preflight validates the exact pinned model checkpoint, data
artifact, source state, precision contract, and output-directory safety.
