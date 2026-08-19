# CUDA / cloud execution

Cloud execution is provider-agnostic. The repository assumes only a Linux host
with an NVIDIA CUDA GPU, a persistent filesystem, the pinned model/data inputs,
and the Python environment from this repository.

## Qualify batching before a serious run

The development reference uses one 2048-token microbatch per optimizer update.
The checked-in CUDA substrate config intentionally preserves that 2,048-token
optimizer batch. GPU memory capacity alone is not evidence for changing it.

On the intended GPU, first run:

```bash
uv sync --extra data --extra eval
make check
make efficiency-cuda-batch-qualification
make select-cuda-batch \
  RESULT=benchmarks/efficiency/results/cuda_batch_qualification.json
```

The qualification measures K=2 MemoryAdd and MemoryTape32 with BF16 autocast,
2048 context, microbatches 1/2/4/8, and `grad_accum_steps=1`. OOM cases are
retained as measurements. The selector chooses the smallest common successful
microbatch reaching at least 90% of each architecture's best throughput across
all of that architecture's feasible tested batches.

If the selected microbatch is 1, the validated 2,048-token optimizer batch is
preserved. If it is larger, the selector explicitly reports that the scientific
optimizer batch changed. Qualify that optimizer-batch change before locking a
core study rather than automatically increasing gradient accumulation.

## Paid-run preflight

Before a paid training run:

```bash
uv run python scripts/cloud_preflight.py --config <config.yaml>
```

The preflight checks source cleanliness, CUDA/BF16 capability, the exact pinned
TinyMistral checkpoint structure/configuration/weights hash, required model and
data paths, data-artifact integrity, checkpoint paths, accidental reuse of an
existing output directory, and reports the config's microbatch and nominal
optimizer-batch token counts.

For a serious core campaign, Phase-A wiring and Phase-B training should use the
same pinned core data artifact (currently `data/dolmino/gpu_2048`). Do not use a
`local_2048` development wiring checkpoint as the parent of the locked core arm.

Keep raw checkpoints, run manifests, and metrics in the relevant benchmark
directory's `results/generated/` path. That subtree is ignored by Git, so keep
it on persistent storage and back it up independently of an ephemeral GPU
instance. Commit only compact result records that are worth retaining.
Provider-specific deployment code is intentionally omitted until it is needed.
