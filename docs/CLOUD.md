# CUDA / cloud execution

Cloud execution is provider-agnostic. The repository assumes only a Linux host
with an NVIDIA CUDA GPU, a persistent filesystem, the pinned model/data inputs,
and the Python environment from this repository.

Before a paid training run:

```bash
uv sync --extra data --extra eval
make check
uv run python scripts/cloud_preflight.py --config <config.yaml>
```

The preflight checks source cleanliness, CUDA/BF16 capability, the exact pinned
TinyMistral checkpoint structure/configuration/weights hash, required model and
data paths, data-artifact integrity, checkpoint paths, and accidental reuse of
an existing output directory.

Keep raw checkpoints, run manifests, and metrics in the relevant benchmark
directory's `results/generated/` path. That subtree is ignored by Git, so keep
it on persistent storage and back it up independently of an ephemeral GPU
instance. Commit only compact result records that are worth retaining.
Provider-specific deployment code is intentionally omitted until it is needed.
