# CUDA / cloud execution

Cloud execution is provider-agnostic. The repository assumes only a Linux host
with an NVIDIA CUDA GPU, a persistent filesystem, the pinned model/data inputs,
and the Python environment from this repository.

Before a paid training run:

```bash
uv sync --extra data --extra eval
uv run pytest -q
uv run python -m compileall -q src scripts tests benchmarks
uv run python scripts/cloud_preflight.py --config <config.yaml>
```

The preflight checks source cleanliness, CUDA/BF16 capability, the exact pinned
TinyMistral checkpoint structure/configuration/weights hash, required model and
data paths, data-artifact integrity, checkpoint paths, and accidental reuse of
an existing output directory.

Keep generated checkpoints and benchmark output in the relevant benchmark
directory's `results/` path. Keep those files on persistent storage and back
them up independently of an ephemeral GPU instance.
Provider-specific deployment code is intentionally omitted until it is needed.
