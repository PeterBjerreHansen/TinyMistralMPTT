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
  RESULT=benchmarks/efficiency/results/generated/cuda_batch_qualification.json
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

For a new on-demand or spot run:

```bash
uv run python scripts/cloud_preflight.py \
  --config <config.yaml> \
  --mode auto \
  --persistent-root /mnt/<persistent-volume>
```

`--mode auto` resolves an empty output directory as a new run and an existing
run with a valid checkpoint as a resume. Ambiguous state fails closed. Use
`--mode new` or `--mode resume` when you want stricter operator intent.

The preflight checks source cleanliness, CUDA/BF16 capability, the exact pinned
TinyMistral checkpoint structure/configuration/weights hash, required model and
data paths, data-artifact integrity, checkpoint paths, recovery state, and the
config's microbatch and nominal optimizer-batch token counts. When a persistent
root is supplied it also verifies that the output directory is underneath it
and reports free space. Once full checkpoints exist, it reserves roughly three
checkpoint sizes so `previous + current + new.tmp` can coexist during an atomic
rotation.

## Spot-safe checkpoint policy

A serious spot run should keep two complete resumable generations and save on
both a token and wall-clock cadence:

```yaml
checkpoint_every_tokens: 500000
checkpoint_every_seconds: 600
checkpoint_keep_last: 2
```

The exact cadence should be finalized after measuring checkpoint I/O on the
selected GPU/storage pair. The intended target is frequent enough that an
arbitrary eviction loses only a small amount of compute while checkpointing
remains a negligible fraction of training time.

Generations live under:

```text
<output_dir>/checkpoints/
  checkpoint_00012001280.pt
  checkpoint_00012500992.pt
  latest.json
```

A new generation is written and fsynced before `latest.json` is advanced and
before the oldest generation is pruned. Auto-resume tries the current generation
first and falls back to the previous one if the newest is unreadable/corrupt.
The checkpoint remains the source of truth for training progress.

`metrics.jsonl` is repaired back to the selected checkpoint on resume, so rows
written after the last durable checkpoint are discarded before replay. The
immutable `run.json` describes the experiment; `segments.jsonl` records each VM
process lifetime and its hardware/source provenance.

The checkpoint format records Git provenance plus deterministic execution-code
and `uv.lock` hashes. Resuming a format-v3 checkpoint from different execution
code or environment is rejected by default. This also permits an identical
source archive to resume a run even when `.git` metadata is unavailable. The
`--allow-source-mismatch` escape hatch is for development only and should not be
used for a locked core campaign.

For the actual training invocation, use the same command on the first VM and
on every replacement VM:

```bash
uv run python scripts/train.py \
  --config <config.yaml> \
  --resume-auto
```

If the output directory is empty, this starts a new run. If a compatible run
exists, it resumes the newest valid generation. A pre-existing run with no
recoverable checkpoint is a hard failure; auto mode never silently starts that
experiment again from token zero.

SIGINT/SIGTERM request a checkpoint at the next completed optimizer boundary,
but spot correctness does not depend on receiving a signal. The hard-failure
model is that the machine may disappear without executing another instruction.

## Scientific snapshots

Resumable checkpoints are operational artifacts and only the newest two are
kept. Optional `snapshot_at_tokens` thresholds write weights-only safetensors
under `<output_dir>/snapshots/` for later scientific analysis. They are never
used by auto-resume.

For a serious core campaign, Phase-A wiring and Phase-B training should use the
same pinned core data artifact (currently `data/dolmino/gpu_2048`). Do not use a
`local_2048` development wiring checkpoint as the parent of the locked core arm.

Keep raw checkpoints, run manifests, metrics, and snapshots on persistent
storage. Generated result subtrees are ignored by Git, so back them up
independently of an ephemeral GPU instance. Commit only compact result records
that are worth retaining.
