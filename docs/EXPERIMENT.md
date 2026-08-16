# Experimental protocol: bootstrap phase

This repository currently implements the architecture-independent vertical
slice only. There is exactly one model variant: `vanilla`.

## What is already fixed

The trainer is token-budget based, uses fixed local token blocks, logs unique
training tokens and token-equivalent backbone compute separately, supports
stateful checkpoint/resume, performs held-out NLL evaluation, and exposes an
optional lm-evaluation-harness adapter.

For vanilla, `effective_passes=1`, so unique tokens and token-equivalent compute
are identical. This counter is already present so future pass schedules do not
change the experiment log schema.

## Phase A / Phase B contract

The infrastructure knows about two phases but does not invent a fake wiring
phase for vanilla:

- **Phase A**: train architecture-added parameters while the TinyMistral
  backbone is frozen. Vanilla has no added parameters, so its Phase A is an
  explicit zero-step no-op.
- **Phase B**: unfreeze/train the full model with the controlled continued-
  pretraining protocol.

The exact FBT Phase-A token budget, learning rate, and pass-loss recipe are
intentionally **not** fixed yet. They should be chosen only after the first FBT
implementation reveals how disruptive its inserted pathway is.

## Resume contract

A training checkpoint contains model and optimizer state, the exact shuffled
block sampler permutation/RNG/position, Python and PyTorch RNG states, token
counters, phase, resolved experiment config, and the SHA-256 of the data
manifest. Resume is rejected if the data manifest has changed.

## Primary and secondary evaluation

Primary metric: held-out next-token NLL/perplexity on the fixed local Dolmino
validation artifact, including a by-source NLL breakdown.

Secondary metric: a checked-in lm-evaluation-harness suite. `quick.yaml` is a
small development battery. `full.yaml` contains the ten 5-shot base-model tasks
used in the Full-bandwidth Transformer evaluation set. Benchmark results are
secondary because a 248M model can be close to chance on several tasks.

## Boundary for the next phase

Do not add sparse memory or a general multipass abstraction yet. The next model
work should implement FBT alone through the existing trainer/data/evaluation
interfaces. Once FBT can train, resume, evaluate, and generate, its actual needs
should determine the minimal common multipass API.
