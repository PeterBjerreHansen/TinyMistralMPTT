# Frozen wired benchmark

This is the immutable Phase-A reference for the first joint co-adaptation
experiments. Both checkpoints were continued from the 262,144-token Phase-A
runs to 1,048,576 unique training tokens with the pretrained TinyMistral
backbone frozen, fixed two-pass training, and pass-loss weights `[0.0, 1.0]`.

Repository commit for the frozen-continuation support: `26e754b`.

## Checkpoints

| Variant | Added parameters | Checkpoint | SHA-256 |
| --- | ---: | --- | --- |
| MemoryAdd | 1,049,600 | `checkpoints/memory_add_frozen_wired_v1.pt` | `62885c820499b987ebf7949b81d6ef0de66e0327f1c9b75d7a865653a046248d` |
| MemoryTape32 | 31,481,856 | `checkpoints/memory_tape32_frozen_wired_v1.pt` | `7799bb01bfbe112309007585bae7e9183b7d01d9df4f2454f640b1c126f9964f` |

## Full validation pass depth

The base pass-1 NLL is `2.66453146` on all 256 validation blocks.

| Pass | MemoryAdd | MemoryTape32 |
| ---: | ---: | ---: |
| 1 | 2.66453146 | 2.66453146 |
| 2 | 2.58310922 | 2.55555622 |
| 3 | 2.59208004 | 2.56346234 |
| 4 | 2.58898712 | 2.55804746 |
| 5 | 2.59043072 | 2.56069112 |
| 6 | 2.59000497 | 2.55911938 |
| 7 | 2.59012520 | 2.55994944 |
| 8 | 2.59008803 | 2.55948577 |

Hidden-state transition RMS decays through depth 8:

```text
MemoryAdd:   0.6370, 0.3727, 0.1533, 0.0795, 0.0310, 0.0114, 0.0039
MemoryTape32: 0.5309, 0.2702, 0.1499, 0.0847, 0.0477, 0.0268, 0.0150
```

## Causal interventions

| Variant | Real memory NLL | Zero memory NLL | Mismatched memory NLL |
| --- | ---: | ---: | ---: |
| MemoryAdd | 2.58310922 | 2.66453146 | 2.70738921 |
| MemoryTape32 | 2.55555622 | 2.66453146 | 2.63280547 |

For MemoryAdd, the non-initial memory residual RMS is `0.02377325` versus
embedding RMS `0.06620123`, a residual-to-embedding ratio of `0.35910589`.

The milestone values in the training logs are based on the periodic 16-block
validation slice. The full 256-block values above are the headline comparison.
