# Backbone learning-rate development

## Question

How much joint backbone adaptation is useful after frozen wiring, and what
conservative pretrained-parameter LR should be used for later development?

## Starting checkpoints

Both architectures start from the immutable wired checkpoints in
`../../../stage1_starting_points/STARTING_POINTS.yaml` for the controlled dose
response.

## Protocol

The recorded sweep uses fixed K=2, pass weights `[0.25, 0.75]`, added-parameter
LR `1e-6`, and pretrained LR in `{0, 3e-8, 1e-7, 3e-7}`. Exact run configs are
preserved under `configs/`.

## Measurements and results

See `RESULTS.md` for 131k pilots, 262k confirmation, one-million-token joint and
frozen continuations, causal interventions, and parameter movement.

## Decision

`1e-7` is the current conservative backbone-LR operating point for Stage 2
development. This is a protocol decision, not a reason to use the final K=2
joint checkpoint as the parent of every later experiment.
