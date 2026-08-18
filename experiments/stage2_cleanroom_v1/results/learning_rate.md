# E2 learning-rate bracket

E2 used 262,144 fresh unique tokens per arm. MemoryAdd and MemoryTape32
started independently from their clean E1 wiring checkpoints. Their added
parameters used `1e-6`; the table varies the backbone learning rate. The
vanilla controls used the same token dose and seed at the indicated rate.

## Memory arms

Full 256-block validation was run at the endpoint with two passes.

| variant | backbone LR | pass-1 NLL | pass-2 NLL |
| --- | ---: | ---: | ---: |
| MemoryAdd | 0 | 2.664531 | 2.571398 |
| MemoryAdd | 3e-8 | 2.662374 | 2.570254 |
| MemoryAdd | 1e-7 | 2.657510 | 2.567687 |
| MemoryAdd | 3e-7 | 2.644916 | 2.561092 |
| MemoryAdd | 1e-6 | 2.611896 | 2.543674 |
| MemoryAdd | 3e-6 | 2.560645 | 2.513007 |
| MemoryAdd | 1e-5 | 2.485316 | 2.461819 |
| MemoryTape32 | 0 | 2.664531 | 2.554481 |
| MemoryTape32 | 3e-8 | 2.662637 | 2.553750 |
| MemoryTape32 | 1e-7 | 2.658384 | 2.552113 |
| MemoryTape32 | 3e-7 | 2.647413 | 2.547850 |
| MemoryTape32 | 1e-6 | 2.617776 | 2.535324 |
| MemoryTape32 | 3e-6 | 2.566183 | 2.509288 |
| MemoryTape32 | 1e-5 | 2.489094 | 2.461305 |

The curve is still improving at `1e-5`, so `1e-5` is an upper-boundary test,
not a selected optimum. The improvement is accompanied by a shrinking
pass-2 memory increment, which means ordinary backbone adaptation explains
part of the gain.

## Vanilla controls

| backbone LR | full-evaluation NLL |
| ---: | ---: |
| 3e-7 | 2.634112 |
| 1e-6 | 2.591856 |
| 3e-6 | 2.537343 |

The vanilla controls confirm that higher backbone learning rates improve the
substrate substantially. Memory results must therefore be compared with the
matched vanilla trajectory, not interpreted from pass-2 NLL alone.

## Memory-use checks

For each candidate, `G_mem = pass-1 NLL - pass-2 NLL`. The mismatched column is
the NLL after replacing the causal memory state with a state from another
sequence.

| variant | backbone LR | G_mem | mismatched NLL |
| --- | ---: | ---: | ---: |
| MemoryAdd | 1e-6 | 0.068222 | 2.651302 |
| MemoryAdd | 3e-6 | 0.047637 | 2.603687 |
| MemoryAdd | 1e-5 | 0.023496 | 2.530608 |
| MemoryTape32 | 1e-6 | 0.082452 | 2.603486 |
| MemoryTape32 | 3e-6 | 0.056895 | 2.566328 |
| MemoryTape32 | 1e-5 | 0.027789 | 2.503233 |

Real memory remains better than zero memory at all three candidate rates, and
mismatched memory is worse than real memory. However, both architectures show
the same trend: the memory-specific advantage falls as the backbone rate
rises. This is evidence for selecting the lowest rate that gives a strong
overall result and stable causal memory use, rather than simply selecting the
lowest pass-2 NLL.

## Protocol status

E2 selected `1e-6` for both the backbone and added parameters. These learning-
rate arms remain protocol-development evidence; no E2 checkpoint is used as a
K-sweep parent. The selected-LR K schedules are recorded separately under
`results/k_sweep.md`, while the final K schedule remains pending context and
efficiency qualification.

The corresponding configs and runs are grouped under:

```text
experiments/stage2_cleanroom_v1/configs/learning_rate/
runs/stage2_cleanroom_v1/learning_rate/
```
