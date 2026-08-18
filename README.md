# TinyMistralMPTT

Research code for MemoryAdd, MemoryTape32, and recurrent inference on a
validated TinyMistral backbone.

## Read first

- `benchmarks/`: controls, historical, development, ad-hoc, core, and efficiency studies.
- `data/`: dataset recipes beside generated data artifacts.
- `benchmarks/**/results/`: experiment-local metrics and reports.
- `docs/`: durable implementation and provenance contracts.
- `src/tiny_mistral/`: validated vendored vanilla implementation.
- `src/tiny_mistral_mptt/`: research models, training, evaluation, and inference.

The checkpoint directory is `checkpoints/TinyMistral-248M-v3/`. This exact name
is retained because it is the upstream model identifier, not this repository's
version. The upstream revision is pinned in `docs/UPSTREAMS.md`.

## Current Stage 2 status

The Stage 2 protocol remains open. E2 selected `1e-6` for both backbone and
added parameters, and the completed 512-token K sweep makes K=2 the provisional
winner within that historical clean-room lineage. The active pre-lock evidence
uses the canonical 2048-token data recipe; the K schedule is intentionally still
pending and no 100M-token main campaign is locked.

The active evidence is documented in:

```text
benchmarks/historical/stage2_cleanroom_v1/results/k_sweep.md
benchmarks/development/k_selection/results/baseline_2048.md
```

Do not start a core Stage 2 run until the development protocol is explicitly
relocked.

## Validate

```bash
uv sync --extra data --extra eval
uv run pytest -q
uv run python -m compileall -q src scripts tests benchmarks
git diff --check
```

Prepare and verify the active 2048-token data artifact:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/verify_data.py data/dolmino/local_2048
```

Run the engineering efficiency battery on the local accelerator with the
targets in `benchmarks/efficiency/`. These measurements are separate from the
scientific experiment record.

The CUDA substrate config and `docs/CLOUD.md` describe the paid-run preflight
path. The cloud preflight validates the exact pinned model checkpoint, data
artifact, source state, precision contract, and output-directory safety.
