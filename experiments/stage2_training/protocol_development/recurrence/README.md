# Recurrent inference development

## Question

Does finite-pass training produce a state transition that can be collapsed to
one-stream online recurrence without large distribution shift?

## Method

The generic evaluator compares:

1. exact K-stream cached incremental inference;
2. collapsed one-stream recurrence after the same K-pass prefill;
3. pass-1 cached vanilla inference from the same checkpoint.

The inference implementation is K-general and uses bounded TinyMistral SWA KV
caches. See `docs/RECURRENT_INFERENCE.md` for the executable contract.

## Current health evidence

For the recently trained K=3 development checkpoints, the reported maximum
exact-versus-recurrent NLL gap was `0.0030` for both architectures. Minimum
hidden cosine was `0.981` for MemoryAdd and `0.997` for MemoryTape32. K=2-trained
checkpoints also tolerated K=3/K=4 inference without observed instability.

These are protocol-development health checks. Future result records should
always state the validation block count, prompt length, continuation length,
checkpoint, and `prefill_passes` sweep explicitly.

## Decision

Recurrent inference is sufficiently healthy to remain a required diagnostic in
Stage 2. It does not by itself determine K or the main-run protocol.
