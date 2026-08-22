# Wired-checkpoint diagnostic baseline

These historical measurements were obtained without optimizer updates on the
completed Stage-2 Recirculation–Tape checkpoint, before recurrent NMP target
normalization was added:

`benchmarks/development/stage_2_local_smoke/results/generated/hybrid_recirculation_smoke/checkpoints/checkpoint_000001048576.pt`

The script evaluated eight pilot training blocks at K=2 on MPS.

| Quantity | Value | Relative to NTP |
| --- | ---: | ---: |
| NTP loss | 1.9425 | 1.00x |
| raw recurrent NMP loss | 11.5173 | 5.93x |
| sparse Tape NMP loss | 0.8609 | 0.44x |
| recurrent target RMS | 19.3580 | — |
| sparse Tape target RMS | 1.6022 | — |

The recurrent target was the adaptive Recirculation source-layer state in this
old raw-state measurement, so its Smooth-L1 scale was much larger than the
post-writer tape target. Current code applies parameter-free RMS normalization
to recurrent NMP by default, so these raw values are not the current expected
diagnostic scale. Re-run the diagnostics after the 10M NTP parent exists; the
recurrent target RMS should then be close to one and the auxiliary coefficient
comparison should be made using the new metrics:

Each report also records target feature standard deviation, per-pass prediction
error RMS, the fraction of Smooth-L1 elements in the linear region, and valid
query/event counts. Use these fields when interpreting a large raw tape target
or a batch whose sparse policy emits no future write.

| Diagnostic setting | Recurrent weight | Tape weight | Weighted NMP / NTP |
| --- | ---: | ---: | ---: |
| low | 0.005 | 0.05 | 5.18% |
| default | 0.01 | 0.10 | 10.36% |
| high | 0.02 | 0.20 | 20.72% |

The new heads are zero-initialized, so pass-1 and pass-2 prediction losses are
identical in this no-update measurement. These diagnostics establish loss
scale, target magnitude, and safe coefficient ranges; they do not measure
learned predictability. Rerun the same three configs after the 10M NTP parent
exists, then use the default setting for the serious continuations unless the
NTP/NMP ratio has moved materially.
