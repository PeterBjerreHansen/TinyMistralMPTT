# Validation gates

This file records durable correctness gates for the codebase. Experimental
results and protocol decisions live under `benchmarks/`.

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

## Benchmark-layout gates

The repository structure is also part of reproducibility:

- historical clean-room result records are retained under
  `benchmarks/historical/stage2_cleanroom_v1/results/`, without runnable configs or
  generated checkpoints;
- reusable substrate and smoke controls live under `benchmarks/controls/`;
- development configs and reports live together under `benchmarks/development/`;
- ad-hoc diagnostics live under `benchmarks/ad_hoc/`;
- core configs and reports live together under `benchmarks/core/` after the
  protocol is explicitly locked;
- generated outputs live in the corresponding experiment `results/` directory;
- active benchmark and data recipes point only to the current checkpoint and
  data namespaces;
- protocol and result records use the same descriptive names as their configs.

These invariants are checked by `tests/test_experiment_layout.py`.

## Standard local gate

```bash
uv run pytest -q
uv run python -m compileall -q src scripts tests benchmarks
```

On Apple hardware also run:

```bash
uv run python scripts/smoke_mps.py
```

## CUDA gate

No CUDA efficiency claim is assumed by the repository. Before a serious GPU
campaign, benchmark the intended models and context lengths and record peak
memory, throughput, parameter count, effective pass count, and inference mode.
