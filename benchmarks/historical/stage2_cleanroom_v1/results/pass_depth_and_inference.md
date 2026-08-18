# Historical K=3 pass depth and inference

> Superseded protocol-development evidence. These runs used backbone LR
> `3e-7`. The related selected-LR K-sweep is summarized separately in
> `k_sweep.md`.

This is the record for the earlier standalone K=3 arms. All four arms use:

- pass weights `[0.05, 0.20, 0.75]`
- backbone learning rate `3e-7`
- added-parameter learning rate `1e-6`
- the corresponding clean E1 wiring checkpoint
- 16-block training validation and 256-block diagnostic validation

## Training endpoints

| variant | token budget | unique tokens | token-equivalent compute | pass-3 NLL |
| --- | --- | ---: | ---: | ---: |
| MemoryAdd | full dose | 1,048,576 | 3,145,728 | 2.430698 |
| MemoryAdd | compute matched | 699,392 | 2,098,176 | 2.442101 |
| MemoryTape32 | full dose | 1,048,576 | 3,145,728 | 2.430992 |
| MemoryTape32 | compute matched | 699,392 | 2,098,176 | 2.440598 |

The endpoint validation above uses 16 blocks, matching the training config.

## Full pass-depth diagnostic

The following values use 256 validation blocks and evaluate passes 1–8. They
remain useful as historical evidence but are not the final K-selection data.

| variant | token budget | pass 2 | pass 3 | pass 8 |
| --- | --- | ---: | ---: | ---: |
| MemoryAdd | full dose | 2.537559 | 2.536231 | 2.539310 |
| MemoryAdd | compute matched | 2.547131 | 2.546151 | 2.548490 |
| MemoryTape32 | full dose | 2.530788 | 2.531983 | 2.531653 |
| MemoryTape32 | compute matched | 2.538348 | 2.540098 | 2.539538 |

Both variants improve sharply from pass 1 to pass 2–3 and remain stable through
pass 8. At the same one-million-token budget, the existing K=2 controls remain
slightly better at pass 8; the compute-matched K=3 controls are worse, as
expected from their lower training dose.

## Memory interventions

These are 256-block evaluations at the final checkpoint.

| variant | token budget | zero memory | real memory | mismatched memory |
| --- | --- | ---: | ---: | ---: |
| MemoryAdd | full dose | 2.619579 | 2.537559 | 2.652405 |
| MemoryAdd | compute matched | 2.630849 | 2.547131 | 2.663711 |
| MemoryTape32 | full dose | 2.625934 | 2.530788 | 2.601586 |
| MemoryTape32 | compute matched | 2.636205 | 2.538348 | 2.609411 |

Real memory beats zero memory in every arm, and mismatched memory is worse than
real memory in every arm. MemoryAdd's residual-to-embedding RMS ratio is
`0.3503` at full dose and `0.3525` at matched compute.

## Bounded recurrent inference

The full-dose checkpoints were evaluated with 32 blocks, a 64-token prompt,
64 continuation tokens, horizons 1–64, and prefill depths 1–4. At prefill K=3
and horizon 64:

| variant | max absolute recurrent/exact NLL gap | recurrent minus vanilla | hidden cosine |
| --- | ---: | ---: | ---: |
| MemoryAdd | 0.001701 | -0.053327 | 0.971837 |
| MemoryTape32 | 0.000405 | -0.057168 | 0.999754 |

Collapsed recurrence remains close to exact finite-K inference and improves on
vanilla in this bounded continuation test. The raw diagnostic JSON files were
written to `/private/tmp/stage2_cleanroom_v2_*_recurrent_32x64.json`.

## Retention

This report is retained as historical evidence only. Its original configs and
checkpoints were removed after the 2048-token qualification became active.
