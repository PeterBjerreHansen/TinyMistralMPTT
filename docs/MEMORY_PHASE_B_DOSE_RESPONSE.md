# Memory Phase-B dose-response

This document records the controlled co-adaptation experiment requested after
the frozen wiring stage. The frozen wired checkpoints were kept immutable and
each Phase-B arm used `init_from`, which gives a fresh optimizer, data stream,
and counter state while preserving the learned wiring.

## Protocol

- Architectures: `memory_add` and `memory_tape32`.
- Source checkpoints:
  - `checkpoints/memory_add_frozen_wired_v1.pt`
    (`sha256=62885c820499b987ebf7949b81d6ef0de66e0327f1c9b75d7a865653a046248d`)
  - `checkpoints/memory_tape32_frozen_wired_v1.pt`
    (`sha256=7799bb01bfbe112309007585bae7e9183b7d01d9df4f2454f640b1c126f9964f`)
- Pilot budget: 131,072 unique tokens per arm.
- Confirmation budget: 262,144 unique tokens per arm.
- Passes during training: `K=2` with loss weights `[0.25, 0.75]`.
- Constant learning rate; added-parameter LR `1e-6`.
- Pretrained-parameter LR dose response: `0`, `3e-8`, `1e-7`, `3e-7`.
- Device/dtype: Apple MPS, float32.
- Full evaluations: 256 validation blocks, depths 1 through 8.

All generated checkpoints and run directories are ignored by git. The configs
and this report are the reproducible experiment record.

## 131k-token pilot

The table reports full-validation NLL at pass 2 and pass 8. The frozen wired
controls are `2.583109` / `2.590088` for MemoryAdd and `2.555556` / `2.559486`
for MemoryTape32.

| variant | pretrained LR | pass 2 NLL | pass 8 NLL | pass-2 gain vs frozen |
|---|---:|---:|---:|---:|
| MemoryAdd | 0 | 2.573722 | 2.581725 | 0.009387 |
| MemoryAdd | 3e-8 | 2.573135 | 2.581124 | 0.009974 |
| MemoryAdd | 1e-7 | 2.571792 | 2.579746 | 0.011318 |
| MemoryAdd | 3e-7 | 2.568172 | 2.576027 | 0.014937 |
| MemoryTape32 | 0 | 2.554995 | 2.559056 | 0.000561 |
| MemoryTape32 | 3e-8 | 2.554627 | 2.558682 | 0.000929 |
| MemoryTape32 | 1e-7 | 2.553786 | 2.557828 | 0.001770 |
| MemoryTape32 | 3e-7 | 2.551511 | 2.555517 | 0.004045 |

The zero-LR rows are important controls: updating only the added parameters
already improves the trained pass-2 loss. The pretrained LR response is
monotonic in this short pilot. `3e-8` is close to the control, while `1e-7`
is the smallest setting with a clear additional gain for both variants.

## 262k-token confirmation at pretrained LR `1e-7`

The selected arms used the same protocol with twice the pilot budget.

| variant | pass 1 NLL | pass 2 NLL | pass 3 NLL | pass 4 NLL | pass 8 NLL |
|---|---:|---:|---:|---:|---:|
| MemoryAdd | 2.657510 | 2.567687 | 2.578563 | 2.574505 | 2.576331 |
| MemoryTape32 | 2.658384 | 2.552113 | 2.560094 | 2.554859 | 2.556259 |

Relative to the frozen wired checkpoints, the selected runs improve pass 2 by
`0.015422` NLL for MemoryAdd and `0.003443` for MemoryTape32. Their pass-8
improvements are `0.013757` and `0.003227`, respectively.

Both models show the same qualitative depth curve: pass 2 is best, pass 3
rebounds slightly, and later passes remain bounded. There is no evidence of
pass-depth divergence in this confirmation.

## Causal memory interventions

These values use the selected 262k-token checkpoints and full validation.

| variant | zero memory | real memory | mismatched memory | real-vs-zero gain | mismatch penalty vs real |
|---|---:|---:|---:|---:|---:|
| MemoryAdd | 2.657510 | 2.567687 | 2.692591 | 0.089823 | 0.124904 |
| MemoryTape32 | 2.658384 | 2.552113 | 2.628538 | 0.106271 | 0.076425 |

Real previous state is therefore materially useful in both architectures, and
the benefit is not explained by simply adding a learned residual independent of
the recurrent state. MemoryAdd's residual-to-embedding RMS ratio is `0.3551`
after co-adaptation (residual RMS `0.023509`, embedding RMS `0.066201`).

## Decision

Use pretrained LR `1e-7` as the conservative Phase-B setting for the next
larger run. Do not introduce `K=3` or multi-depth training yet: the current
evidence supports extending the selected `K=2` run first, then repeating the
depth and intervention checks at the larger budget.
