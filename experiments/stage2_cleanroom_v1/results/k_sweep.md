# Selected-LR K sweep

This is the active final K-schedule comparison after E2 selected a common
backbone and added-parameter learning rate of `1e-6`.

All eight arms start independently from the clean E1 wiring checkpoint for
their architecture and use 1,048,576 unique tokens, sequence length 512,
seed 1337, and the same optimizer settings. K=3 uses weights
`[0.05, 0.20, 0.75]`; K=2 uses `[0.25, 0.75]`; mixed schedules use the
corresponding K-specific weights.

| arm | schedule | expected compute |
| --- | --- | ---: |
| K2 | K=2 | 2.0x |
| K2/K3 90/10 | P(K=2)=0.9, P(K=3)=0.1 | 2.1x |
| K2/K3 50/50 | P(K=2)=0.5, P(K=3)=0.5 | 2.5x |
| K3 | K=3 | 3.0x |

The final choice will use common full validation, realized K histograms and
token-equivalent compute, pass-depth stability, real/zero/mismatched memory
interventions, and bounded recurrent-inference checks. Until that comparison
is complete, the K schedule and recurrent inference depth remain pending.

Configs:

```text
experiments/stage2_cleanroom_v1/configs/k_sweep/memory_add/
experiments/stage2_cleanroom_v1/configs/k_sweep/memory_tape32/
```
