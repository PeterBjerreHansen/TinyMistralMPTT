# Sparse write-frequency study

This planned study is the first scientific use of `SparseMemoryTape`. It is
intentionally **not locked yet**. The configs define the first sequential
cadence campaign, but generated checkpoints and results remain outside Git.

The order is:

```text
Phase-A wiring at W=32
    -> short Phase-B cadence screen at 262,144 tokens
    -> continue Dense/C1/best sparse cadences to 1,048,576 tokens
    -> sweep W for the selected cadence
    -> compare the selected SparseTape with MemoryAddSparseTape
```

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

Every sparse config explicitly sets `memory_write_mode: periodic`, its
`memory_write_stride`, and `memory_window: 32`. Token-triggered writes are not
part of this study.

The C=1 arm is the learned-writer control for dense Tape32. The writer is
identity-initialized and the bridge is numerically tight before training, so the
C sweep separates learned write transformation from sparse write cadence.

Do not sweep `W` in this study. After selecting a useful cadence, create a
separate window study (for example W=8/16/32/64). Only after selecting a sparse
tape should the `MemoryAddSparseTape` hybrid be compared against its two
components.

Generated run artifacts belong under `results/generated/` once arms are
materialized and remain ignored by Git.

## Execution order

Run Phase A first:

```bash
uv run python scripts/verify_study.py benchmarks/development/sparse_write_frequency
uv run python scripts/train.py --config benchmarks/development/sparse_write_frequency/phase_a_dense.yaml
uv run python scripts/train.py --config benchmarks/development/sparse_write_frequency/phase_a_c1.yaml
uv run python scripts/train.py --config benchmarks/development/sparse_write_frequency/phase_a_c4.yaml
uv run python scripts/train.py --config benchmarks/development/sparse_write_frequency/phase_a_c8.yaml
uv run python scripts/train.py --config benchmarks/development/sparse_write_frequency/phase_a_c16.yaml
uv run python scripts/train.py --config benchmarks/development/sparse_write_frequency/phase_a_c32.yaml
```

At each 1,048,576-token wiring endpoint, inspect pass-depth NLL, memory
interventions, writer diagnostics, finite values, and the Phase-A gradient
records. Stop a cadence that shows no sequence-specific memory signal rather
than changing its learning rate.

Then run the short Phase-B screen. The configs stop at 262,144 tokens; continue
only Dense, C1, and the best one or two genuinely sparse cadences to 1,048,576
tokens after reviewing the diagnostics.

The campaign keeps K=2, context 2048, optimizer batch 2048 tokens/update,
seed 1337, architecture seed 4242, periodic writing, and the validated Phase-A
and Phase-B learning rates fixed. It does not tune C, W, K, pass weights, or
writer learning rate simultaneously.
