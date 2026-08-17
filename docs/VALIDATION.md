# Validation record

This document records durable validation gates for the current codebase. Detailed
training-campaign results live under `experiments/` so this file does not become
a chronological experiment notebook.

## Vanilla substrate

The validated backbone targets `M4-ai/TinyMistral-248M-v3` and preserves its
checkpoint hierarchy, GQA/local-attention math, cache semantics, and generation
contract. Provenance and the one documented non-numerical dispatch hardening are
recorded in `UPSTREAMS.md`; `VANILLA_SOURCE.sha256` guards the vendored source.

The original Mac bootstrap established:

- 248,024,064 checkpoint parameters;
- checkpoint weights SHA-256
  `9432ee6e0681473a9ed513e43362d9911832f9a5c7faded76f46ec66c55a9d3b`;
- exact MPS local-window forward parity in the smoke test;
- deterministic Dolmino development data with 1,048,576 train tokens and
  131,072 validation tokens at sequence length 512;
- base held-out NLL `2.66453146`;
- vanilla continued-pretraining held-out NLL `2.61463786` after 262,144 tokens.

## Multipass contracts

The test suite enforces the properties that can silently invalidate an
experiment:

- pass 1 of FBT, MemoryAdd, and MemoryTape32 is exact vanilla;
- MemoryAdd is an exact all-depth fixed point at zero initialization;
- previous-pass feedback is shifted by exactly one token for one-state models;
- MemoryTape32 full-sequence attention is strict-past and local;
- Phase A freezes pretrained parameters and Phase B restores backbone gradients;
- pass-loss weights are right-aligned and normalized;
- pass-count sampling is deterministic and checkpointed;
- `resume_from` restores trajectory state while `init_from` loads model weights
  only;
- exact K-stream cached inference matches full finite-pass recomputation for
  MemoryAdd and MemoryTape32 at multiple K;
- K=1 cached inference is the vanilla boundary and requires no feedback hooks;
- unsupported K>1 cached feedback is rejected explicitly rather than failing
  through a variant `NotImplementedError`;
- recurrent K>1 handoff is seeded from pass K-1 and matches the exact path for
  the first processed continuation token;
- snapshot-before-update semantics prevent same-position feedback leakage;
- MemoryAdd retains one feedback vector and MemoryTape32 retains a bounded,
  ordered ring;
- MemoryTape32's cached bank reader accepts exactly one query token;
- cache positions remain correct beyond TinyMistral's sliding-attention window.

## Memory campaign evidence

MemoryAdd and MemoryTape32 both completed mature frozen-backbone wiring and a
controlled Phase-B backbone-LR dose response on the target Mac. The exact
configs, checkpoint hashes, pass-depth curves, interventions, parameter
movement, and matched frozen controls are preserved under
`../experiments/memory_phase_b/`.

At the final recorded joint checkpoints (K=2 training, pretrained LR `1e-7`,
added LR `1e-6`), full 256-block validation was:

```text
MemoryAdd:     pass 1 2.6394, pass 2 2.5541, pass 8 2.5638
MemoryTape32:  pass 1 2.6426, pass 2 2.5432, pass 8 2.5473
```

Real/zero/mismatched interventions remained state-sensitive. See
`../experiments/memory_phase_b/DOSE_RESPONSE.md` for the complete record.

FBT retrofit pilots, including the retired calibrated-initialization experiment,
are isolated under `../experiments/fbt_retrofit/`.

## Current cleanup acceptance

The cleanup candidate was exercised on the Linux/CPU reference environment with:

```text
129 passed, 8 skipped
```

The eight skips are MPS-only tests. The same source also passes:

```bash
python -m compileall -q src scripts tests experiments
```

All eleven canonical `configs/mac/*.yaml` files parse and validate through
`ExperimentConfig`. Historical sweep configs are intentionally kept under
`experiments/`; the calibrated-FBT YAML is marked historical because its one-off
config fields were retired from the stable API.

## Target-Mac gate

Before interpreting new recurrent results from this cleanup revision, rerun:

```bash
uv run pytest -q
uv run python -m compileall -q src scripts tests experiments
```

Then evaluate the preserved final checkpoints with the canonical architecture
configs, for example:

```bash
uv run python scripts/eval_recurrent.py \
  --config configs/mac/memory_add_phase_b.yaml \
  --checkpoint <memory-add-checkpoint.pt> \
  --prefill-passes 2 \
  --prompt-tokens 256 \
  --continuation-tokens 256

uv run python scripts/eval_recurrent.py \
  --config configs/mac/memory_tape32_phase_b.yaml \
  --checkpoint <memory-tape32-checkpoint.pt> \
  --prefill-passes 2 \
  --prompt-tokens 256 \
  --continuation-tokens 256
```

K=2 is the first scientific comparison because the recorded checkpoints were
trained with fixed K=2. Other positive prefill depths are inference-depth
ablations, not claims that those depths were optimized during training.

## CUDA gate

No CUDA efficiency claim is validated yet. Before a rented-GPU run, benchmark
vanilla, MemoryAdd, FBT, and MemoryTape32 at the intended context length and
record peak memory, throughput, parameter count, and effective pass count.
