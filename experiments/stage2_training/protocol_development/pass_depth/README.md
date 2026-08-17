# Pass-depth development

## Question

Does training with K=3 remain stable and useful enough to justify deeper
supervision and roughly 1.5x the backbone-pass compute of K=2?

## Starting checkpoints

The completed 262k K=3 development runs were initialized from the mature
one-million-token K=2 jointly adapted checkpoints. This makes them useful health
experiments, but **not a controlled estimate of the K=2 versus K=3 causal
effect**: they also received additional training and changed loss weights.

## Protocol

Both K=3 runs use pretrained LR `1e-7`, added LR `1e-6`, constant schedule,
fixed `P(K=3)=1`, and pass weights `[0.1, 0.3, 0.6]`. The `*_262k.yaml` files
record the completed short runs. The `*_continue_1m.yaml` files preserve the
then-proposed exact trajectory continuation; they are development configs, not
locked main-run configs.

## Results

On full 256-block validation after the 262k K=3 continuation:

| model | mature K=2 pass 2 | K=3 run pass 2 | pass 3 | pass 8 |
|---|---:|---:|---:|---:|
| MemoryAdd | 2.5541 | 2.5518 | 2.5536 | 2.5557 |
| MemoryTape32 | 2.5432 | 2.5408 | 2.5460 | 2.5436 |

Causal interventions remained strong:

- MemoryAdd: real `2.5518`, zero `2.6351`, mismatched `2.6659`; residual /
  embedding RMS `0.347`.
- MemoryTape32: real `2.5408`, zero `2.6388`, mismatched `2.6145`.

The third pass was not the best prediction pass, so K=3 training should not be
conflated with K=3 being the preferred inference depth.

## Decision

K=3 is viable and informative. Before locking Stage 2, compare K under a
compute-aware controlled design from the selected Stage 1 starting points. Do
not promote a K=3 development descendant solely because it is newer.
