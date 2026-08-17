# Capability ablations

This directory is reserved for downstream questions after the Stage 2 training
protocol is locked and the principal models are trained.

Likely questions include:

- long-horizon recurrent-memory capability rather than only 512-token LM NLL;
- genuinely sparse/landmark long-range memory and memory-stride sweeps;
- eventual fast-recurrence + sparse-memory hybrid;
- dense-memory quality/compute upper comparisons if useful.

These are not declared a third training stage yet. New architecture work should
be added only when it answers one of these capability questions cleanly.
