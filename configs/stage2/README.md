# Stage 2 protocol pending

Stage 2 is open for protocol development. E2 selected a common Phase-B
backbone and added-parameter learning rate of `1e-6`, but no K schedule or
recurrent inference depth is canonical yet.

The active comparison is the eight-arm sweep under:

```text
experiments/stage2_cleanroom_v1/configs/k_sweep/
```

Do not start a main Stage 2 run from this directory until the K schedule is
selected and the protocol is relocked.
