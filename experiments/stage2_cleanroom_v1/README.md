# Stage 2 clean-room lineage

This directory contains the independent evidence used to select the locked
Stage 2 protocol. The starting points, hashes, protocol, and results are in
`PROTOCOL.yaml` and `results/`.

## Completed arms

- `wiring/`: fresh MemoryAdd and MemoryTape32 wiring checkpoints.
- `learning_rate/`: independent backbone learning-rate arms.
- `pass_depth/`: K=2, K=3, and compute-matched K=3 controls.
- `mixtures/`: optional K=2/K=3 schedules with K-specific loss weights.

Each config has a matching generated run directory under
`runs/stage2_cleanroom_v1/`.

## Locked choice

The promoted Stage 2 configs use K=3, pass weights `[0.05, 0.20, 0.75]`, backbone
learning rate `3e-7`, added-parameter learning rate `1e-6`, and 1,048,576
unique tokens. K=2 and compute-matched K=3 remain controls.

The mixed-K runs are diagnostics only. They sample K independently per
microbatch, record realized histograms and compute, and preserve scheduler/RNG
state in their checkpoints.
