# Sparse write-frequency study

This planned study is the first scientific use of `SparseMemoryTape`. It is
intentionally **not locked yet**: the CUDA qualification for the existing dense
Add/Tape baseline should finish first, and each sparse cadence needs its own
Phase-A wiring checkpoint before Phase-B comparison configs are frozen.

The first controlled sweep should keep `memory_window: 32` fixed and vary only
the tape write process:

| intended arm | write stride C | reader candidates W | interpretation |
| --- | ---: | ---: | --- |
| dense Tape32 | 1 raw state/token | 32 | existing dense control |
| sparse C1/W32 | 1 | 32 | learned-writer control; isolates writer effect |
| sparse C4/W32 | 4 | 32 | quarter-rate writes |
| sparse C8/W32 | 8 | 32 | eighth-rate writes |
| sparse C16/W32 | 16 | 32 | sixteenth-rate writes |
| sparse C32/W32 | 32 | 32 | thirty-second-rate writes |

`C=1` is a hard architectural bridge: with the identity-initialized writer and
matching reader weights, SparseMemoryTape is exactly equivalent to dense
MemoryTape32 before writer training. The C sweep therefore separates learned
write transformation from sparse write cadence.

Do not sweep `W` in this study. After selecting a useful cadence, create a
separate window study (for example W=8/16/32/64). Only after selecting a sparse
tape should the `MemoryAddSparseTape` hybrid be compared against its two
components.

Generated run artifacts belong under `results/generated/` once arms are
materialized and remain ignored by Git.
