> **Record status.** This file preserves the LR-development campaign and the
> decisions made at the time. It is Stage 2 protocol-development evidence, not
> the current locked main protocol. The authoritative lock status is
> `../../main/LOCKED_PROTOCOL.md`.

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

## One-million-token Phase-B continuation

The selected `1e-7` arms were then resumed with `resume_from` to a Phase-B
budget of 1,048,576 unique tokens. Matched frozen controls were resumed from
the 131,072-token `pretrained_lr=0` pilot and used the same budget, schedule,
pass weights, and added-parameter LR. The joint arms were resumed from their
262,144-token selected checkpoints. No K=3 or multi-depth training was added.

The table reports full 256-block validation NLL at pass 2 and pass 8. These
are the preserved milestone checkpoints, not the 16-block training logs.

| variant | backbone LR | 524k pass 2 / 8 | 786k pass 2 / 8 | 1.049M pass 2 / 8 |
|---|---:|---:|---:|---:|
| MemoryAdd | `1e-7` | 2.561891 / 2.570848 | 2.557571 / 2.567263 | 2.554071 / 2.563780 |
| MemoryAdd | `0` | 2.569067 / 2.578236 | 2.568023 / 2.578295 | 2.567518 / 2.577851 |
| MemoryTape32 | `1e-7` | 2.548911 / 2.553071 | 2.545950 / 2.550083 | 2.543211 / 2.547322 |
| MemoryTape32 | `0` | 2.553428 / 2.557666 | 2.552428 / 2.556687 | 2.551509 / 2.555816 |

At the final endpoint, joint co-adaptation improves pass 2 over the frozen
control by `0.013447` NLL for MemoryAdd and `0.008298` for MemoryTape32. The
same gaps at pass 8 are `0.014071` and `0.008494`. The result confirms the
earlier interpretation: MemoryAdd is more adaptation-limited, while Tape32
gets closer to its frozen-backbone ceiling.

The joint runs also improve ordinary pass 1. On full validation, MemoryAdd
pass 1 is `2.639357` versus `2.664531` for its frozen control, and Tape32 is
`2.642569` versus `2.664531`. The frozen controls retain the original vanilla
pass-1 value because their pretrained parameters are held fixed.

## Final causal interventions

The following values are from the final 1.049M checkpoints and full validation.

| variant | backbone LR | zero memory | real memory | mismatched memory | real-vs-zero gain |
|---|---:|---:|---:|---:|---:|
| MemoryAdd | `1e-7` | 2.639357 | 2.554071 | 2.670455 | 0.085285 |
| MemoryAdd | `0` | 2.664531 | 2.567518 | 2.693403 | 0.097013 |
| MemoryTape32 | `1e-7` | 2.642569 | 2.543211 | 2.618965 | 0.099358 |
| MemoryTape32 | `0` | 2.664531 | 2.551509 | 2.632321 | 0.113022 |

Real state remains best in every condition. MemoryAdd's mismatched state is
worse than zero state, while Tape32's mismatched state is better than zero but
still substantially worse than correctly aligned state. This is consistent
with a compact sequence-specific correction for MemoryAdd and a more tolerant,
addressable reader for Tape32.

MemoryAdd's final residual-to-embedding RMS ratio is `0.348708` for the joint
run and `0.351803` for the frozen control. The residual remains a correction,
not a replacement of the token representation.

## Parameter movement from the mature wired checkpoint

For each final checkpoint, movement is measured as
`||theta - theta_0||_2 / ||theta_0||_2`, separately for the pretrained
backbone and added parameters.

| variant | backbone LR | backbone relative movement | added relative movement |
|---|---:|---:|---:|
| MemoryAdd | `1e-7` | `1.473e-4` | `8.153e-4` |
| MemoryAdd | `0` | `0` | `8.139e-4` |
| MemoryTape32 | `1e-7` | `1.410e-4` | `1.770e-3` |
| MemoryTape32 | `0` | `0` | `1.870e-3` |

The backbone moves only about 0.014% in relative L2 norm while the added
pathway moves more. This gives quantitative support for the claim that the
recurrent interface is incorporated with minimal disruption to TinyMistral.

The final decomposition is therefore:

| variant | mature wired pass-2 | frozen-control pass-2 | joint pass-2 |
|---|---:|---:|---:|
| MemoryAdd | 2.583109 | 2.567518 | 2.554071 |
| MemoryTape32 | 2.555556 | 2.551509 | 2.543211 |

For MemoryAdd, continued added-pathway optimization accounts for about half of
the total pass-2 gain and backbone co-adaptation supplies the other half. For
Tape32, the added pathway is closer to saturation, so the relative share from
backbone co-adaptation is larger even though the absolute improvement is
smaller.

The next safe step remains to keep K=2 and the `1e-7` operating point as the
mainline setting. `3e-7` should remain a later LR ablation rather than being
introduced together with a new pass-depth objective.
