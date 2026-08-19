# Validation gates

This file records durable correctness gates for the codebase. Experimental
results and protocol decisions live under `benchmarks/`; reusable diagnostics
remain executable from `scripts/` rather than becoming benchmark studies by
default.

## Vanilla substrate

The validated backbone targets `M4-ai/TinyMistral-248M-v3`. Provenance and the
single documented non-numerical dispatch hardening are recorded in
`UPSTREAMS.md`; `VANILLA_SOURCE.sha256` guards the vendored source.

The substrate tests cover model loading, reference/local attention parity,
sliding-window masks, cached decoding, generation, device handling, and source
manifest integrity.

## Multipass architecture gates

The suite must continue to enforce:

- pass 1 of every multipass variant is exact vanilla;
- MemoryAdd is an all-depth vanilla fixed point at zero initialization;
- one-state previous-pass feedback is shifted by exactly one token;
- MemoryTape32 full-sequence reads are strict-past and locally bounded;
- Phase A freezes pretrained parameters and Phase B restores backbone gradients;
- pass-loss weights are right-aligned and normalized;
- pass-count sampling is deterministic and checkpointed;
- `init_from` loads weights only while `resume_from` restores the exact training
  trajectory.

## Cached inference gates

For MemoryAdd and MemoryTape32:

- exact K-stream cached inference matches full finite-pass recomputation for
  multiple K values;
- K=1 exact and recurrent inference is the vanilla cached boundary;
- unsupported K>1 cached feedback is rejected explicitly;
- recurrent K>1 is seeded from pass K-1 and the first processed continuation
  token matches the exact path;
- snapshot-before-update prevents same-position feedback leakage;
- MemoryAdd retains one feedback vector;
- MemoryTape32 retains an ordered bounded ring and its cached bank reader accepts
  exactly one query token;
- absolute cache positions remain correct beyond the self-attention sliding
  window.

Pass-depth stability, memory interventions, and exact-vs-recurrent drift are
validation measurements available for any checkpoint. They should become a
retained benchmark result only when they support an actual scientific decision.

## Benchmark and study gates

Repository organization follows semantic rather than filename-only invariants:

- runnable configs live with the control or study that owns them;
- development/core studies use `STUDY.yaml` instead of a central config tree;
- the manifest names the scientific question and comparison axes but does not
  duplicate execution settings from runnable configs;
- every runnable config in a study is declared by the manifest;
- a study arm writes raw run artifacts only beneath its local
  `results/generated/<arm>/` directory;
- compared arms must match on every config field except the arm-local output
  path, declared `experimental_axes`, and any explicitly documented
  `allowed_differences`;
- historical reports are lightweight read-only evidence rather than an active
  runnable surface;
- current configs may not depend on historical namespaces.

`tests/test_experiment_layout.py`, `tests/test_studies.py`, and
`scripts/verify_study.py` enforce these conventions.

## Standard local gate

```bash
make check
```

This runs pytest, byte-compilation, study verification, and `git diff --check`
when Git metadata is available. On Apple hardware also run:

```bash
uv run python scripts/smoke_mps.py
```

## CUDA gate

No CUDA efficiency claim or large optimizer batch is assumed by the repository.
The 2048-context development reference uses 2,048 unique tokens per optimizer
update. Before a serious GPU campaign, run the K=2 CUDA batch-qualification
suite at `grad_accum_steps=1`, record OOM/throughput/peak-memory behavior, and
select the smallest common efficient MemoryAdd/MemoryTape32 microbatch. A
selected batch above 1 is a protocol change, not merely a hardware setting, and
must be qualified before a core study is locked.

The efficiency runner reports microbatch tokens, gradient accumulation,
optimizer-batch tokens, optimizer-step throughput, unique-token throughput, and
memory telemetry. The trainer records the same batching semantics in `run.json`
and per-update metrics. Finally run `scripts/cloud_preflight.py` against the
exact intended paid-run config.

### Sparse-memory gates

The sparse-memory branch additionally requires:

- C=1 identity-writer equivalence to dense MemoryTape32;
- strict read-before-write visibility for periodic and token triggers;
- record-count (not source-token-distance) window semantics;
- masked empty sparse banks with exact-zero, finite output;
- Phase-A gradients through writer and readers with a frozen backbone;
- exact cached/full-prefix agreement for SparseMemoryTape and the hybrid;
- hybrid decomposition into its MemoryAdd and SparseTape components;
- bounded sparse cached banks and per-example token-triggered writes.
