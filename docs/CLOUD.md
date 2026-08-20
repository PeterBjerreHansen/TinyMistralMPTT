# CUDA / cloud execution

Cloud execution is provider-agnostic. A serious run assumes a Linux CUDA host,
a persistent filesystem, the pinned model/data inputs, and this repository's
locked Python environment.

## Qualify batching first

The 2048-linguistic-token development reference uses `batch_size=1` and
`grad_accum_steps=1`. GPU capacity alone is not evidence for changing the
optimizer batch.

On the intended GPU:

```bash
uv sync --extra data --extra eval
make check
make efficiency-cuda-batch-qualification
make select-cuda-batch \
  RESULT=benchmarks/efficiency/results/generated/cuda_batch_qualification.json
```

The qualification compares K=2 MemoryAdd and dense Tape with BF16 autocast and
accumulation 1 over microbatches 1/2/4/8. OOM cases are retained. The selector
chooses the smallest common successful batch reaching the configured efficiency
threshold for both architectures. If this is larger than 1, the scientific
optimizer batch has changed and must be qualified before a core study is locked.

For memory-token models, benchmark output distinguishes linguistic sequence
length from expanded physical model positions.

## Paid-run preflight

Use the same config and persistent volume that the training process will use:

```bash
uv run python scripts/cloud_preflight.py \
  --config <config.yaml> \
  --mode auto \
  --persistent-root /mnt/<persistent-volume>
```

`--mode auto` resolves an empty output directory as a new run and a compatible
existing run as resume. Existing ambiguous trajectory artifacts fail closed.
`--mode new` and `--mode resume` are available when operator intent should be
explicit.

The preflight checks:

- CUDA/BF16 capability;
- the pinned TinyMistral checkpoint and data artifact integrity;
- source cleanliness and execution-code/`uv.lock` provenance;
- run/checkpoint compatibility and recoverability;
- output directory containment beneath the persistent root;
- free disk space for durable checkpoint rotation;
- linguistic versus physical batching quantities for memory-token runs;
- the exact tape write/visibility configuration.

## Spot-safe checkpoint policy

A serious spot run should retain two complete resumable generations and save on
both a token and wall-clock cadence, for example:

```yaml
checkpoint_every_tokens: 500000
checkpoint_every_seconds: 600
checkpoint_keep_last: 2
```

The cadence should be measured on the selected GPU/storage pair. The goal is to
bound lost compute while keeping checkpoint I/O a small fraction of wall time.

Generation state is:

```text
<output_dir>/checkpoints/
  checkpoint_00012001280.pt
  checkpoint_00012500992.pt
  latest.json
```

A new checkpoint is written to a temporary file, flushed/fsynced, atomically
renamed, reopened for validation, and only then advertised by `latest.json`.
The oldest generation is pruned only after the new pointer is durable. Auto
resume tries the current generation and falls back to the previous valid one if
the newest is unreadable.

The checkpoint is the trajectory source of truth. On resume, `metrics.jsonl` is
repaired back to the selected checkpoint so metric rows written after durable
training progress are discarded.

## Run provenance

`run.json` is the immutable experiment description. `segments.jsonl` records
each process lifetime, parent checkpoint, source identity, and hardware. Resume
checks deterministic execution-code and `uv.lock` hashes; Git metadata is useful
when present but an identical source archive can still be identified by content
hash.

`--allow-source-mismatch` is a development escape hatch and should not be used
for a locked campaign.

## Training invocation

Use the same command on the first and replacement instances:

```bash
uv run python scripts/train.py \
  --config <config.yaml> \
  --resume-auto
```

If the run directory is truly new, this starts from zero. Otherwise it resumes
the newest valid generation. A pre-existing run without a recoverable
checkpoint is a hard failure; auto mode never silently restarts it.

SIGINT/SIGTERM request a checkpoint at the next completed optimizer boundary.
Correctness does not depend on receiving a signal: the hard failure model is an
instance that disappears without executing another instruction.

## Scientific snapshots

Resumable checkpoints are operational and only the newest configured generations
are retained. Optional `snapshot_at_tokens` writes weights-only safetensors to
`<output_dir>/snapshots/` for analysis. They are never used by auto-resume.

For a locked campaign, Phase-A initialization and Phase-B training should use
the same pinned core data artifact as the reported validation split. Keep the
entire generated run directory on persistent storage and back it up separately
from an ephemeral instance.
