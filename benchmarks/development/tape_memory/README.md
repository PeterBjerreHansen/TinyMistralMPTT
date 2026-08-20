# Tape-memory development study

This is the clean-break study for the unified `tape` architecture. The runnable
arms are intentionally not checked in until the tape/MEM substrate has passed
its semantic, cached-inference, interruption-recovery, and CUDA-attention gates.

Planned order:

1. dense tape wiring;
2. periodic C4/C8/C16/C32 at fixed W=32;
3. select a cadence;
4. compare periodic, `<MEM>` visible, and `<MEM>` write-only at that cadence;
5. compare the selected `tape` against `memory_add` and `tape_add_hybrid`.

Dense tape is the C=1 endpoint of the same writer/reader implementation, so a
separate periodic-C1 scientific arm is unnecessary beyond equivalence tests.
