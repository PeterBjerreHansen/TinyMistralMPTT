# Training contract

This document defines reusable trainer semantics. It deliberately does not say
which Stage 2 protocol is currently preferred; those decisions live under
`experiments/` and, once locked, under `configs/stage2/`.

## Research stages versus trainer phases

The project has two research stages:

- **Stage 1:** construct/select viable architecture-specific starting points.
- **Stage 2:** develop and then lock the training protocol for the comparison.

The trainer independently has two mechanics:

- **Phase A:** pretrained TinyMistral parameters are frozen; only added
  architecture parameters train.
- **Phase B:** the full model is differentiable, with independent pretrained and
  added-parameter learning rates.

A Stage 1 experiment normally uses Phase A. Stage 2 normally uses Phase B, but
the names are intentionally not interchangeable.

## Pass objective

Given pass losses `L_1 ... L_K`, configured non-negative loss weights are
right-aligned to the sampled pass count and normalized to sum to one:

```text
L = sum_k w_k L_k
```

Examples:

```yaml
pass_loss_weights: [0.0, 1.0]
pass_loss_weights: [0.25, 0.75]
pass_loss_weights: [0.1, 0.3, 0.6]
```

`null` means uniform weighting.

## Pass-count schedule

Pass count and loss weighting are separate controls. `pass_schedule` is a
stateful token-indexed schedule whose stages define a probability distribution
over positive K values:

```yaml
pass_schedule:
  - probabilities:
      2: 1.0
```

or, if a future experiment explicitly requires a mixture:

```yaml
pass_schedule:
  - probabilities:
      1: 0.50
      2: 0.45
      3: 0.05
```

The scheduler has an independent RNG and checkpointed state. Fixed-K training
is therefore a protocol choice, not an implementation limit.

## Parameter groups and learning-rate schedules

Phase B maintains separate optimizer groups for pretrained and architecture-
added parameters. Each group has its own base LR; the configured LR schedule
supplies a common multiplier. Supported schedules are `constant`, `cosine`, and
`piecewise_linear`.

## `init_from` versus `resume_from`

These operations have intentionally different scientific meanings.

### `init_from`

Loads model parameters only. Optimizer state, sampler state, pass-scheduler
state, RNG streams, and counters start fresh. Use this when starting a new
experimental stage or arm from a selected checkpoint.

### `resume_from`

Restores the exact training trajectory: model, optimizer, sampler, pass
scheduler, RNG state, counters, phase, and compatible experiment config. Use it
only to continue the same run.

A development checkpoint should never become a main-run parent merely because
it is newest. Main Stage 2 ancestry is defined by the locked protocol and the
Stage 1 starting-point manifest.

## Token accounting

For a microbatch with N token IDs and K backbone passes:

```text
unique_tokens_seen += N
token_equivalent_compute += N * K
```

Both should be reported when comparing different pass depths.

## Evaluation during training

Small periodic validation slices are health diagnostics, not headline results.
Full held-out evaluations should be run from preserved checkpoints. During a
locked main run, diagnostics are read-only: they describe the trajectory and do
not trigger architecture-specific LR or objective changes except for genuine
engineering failure.
