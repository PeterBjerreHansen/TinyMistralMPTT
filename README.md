# TinyMistralMPTT

Research code for MemoryAdd, MemoryTape32, and recurrent inference on a
validated TinyMistral backbone.

## Repository map

- `src/tiny_mistral/`: validated vendored vanilla TinyMistral implementation.
- `src/tiny_mistral_mptt/`: research architectures, training, evaluation, and inference.
- `benchmarks/`: scientific controls, development studies, historical evidence,
  decisive core studies, ad-hoc work, and engineering efficiency measurements.
- `data/`: dataset recipes beside the generated artifacts they define.
- `evaluation/`: reusable evaluation-suite definitions.
- `docs/`: durable architecture, training, inference, precision, data, and
  provenance contracts.

There is intentionally no central `configs/` directory. Runnable settings live
with the study or asset that owns them. Development/core studies use a
`STUDY.yaml` for the scientific question and arm membership; the runnable YAML
files remain the sole source of execution parameters.

Within a benchmark study, compact result records belong directly in `results/`.
Raw checkpoints, `run.json`, `metrics.jsonl`, and similar training artifacts go
under `results/generated/`, which is ignored by Git.

The checkpoint directory is `checkpoints/TinyMistral-248M-v3/`. This exact name
is retained because it is the upstream model identifier, not this repository's
version. The upstream revision is pinned in `docs/UPSTREAMS.md`.

## Current Stage 2 status

The Stage 2 protocol remains open. The historical clean-room lineage selected
`1e-6` for both backbone and added parameters and provisionally favored K=2 at
512-token context. The active 2048-token qualification also supports K=2 as the
lower-compute baseline, but no 100M-token core campaign is locked.

Current evidence:

```text
benchmarks/historical/stage2_cleanroom_v1/results/k_sweep.md
benchmarks/development/k_selection/results/baseline_2048.md
```

Do not start a core Stage 2 campaign until a core study is explicitly declared
and locked.

## Validate

```bash
uv sync --extra data --extra eval
make check
```

`make check` runs tests, byte-compilation, study-manifest verification, and
`git diff --check` when the checkout has Git metadata.

Prepare and verify the active local 2048-token data artifact:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/verify_data.py data/dolmino/local_2048
```

Run engineering efficiency batteries using the targets documented in
`benchmarks/efficiency/`. Before paid CUDA training, follow `docs/CLOUD.md` and
run the provider-agnostic cloud preflight against the exact intended config.
