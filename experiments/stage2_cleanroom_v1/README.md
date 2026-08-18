# Stage 2 clean-room lineage

This directory contains the independent evidence used to select the locked
Stage 2 protocol. The starting points, hashes, protocol, and results are in
`PROTOCOL.yaml` and `results/`.

## Evidence layout

- `wiring/`: fresh MemoryAdd and MemoryTape32 wiring checkpoints.
- `learning_rate/`: independent backbone learning-rate arms.
- `pass_depth/`: historical K=2/K=3 arms at the superseded 3e-7 backbone LR.
- `mixtures/`: historical mixed-K arms at the superseded 3e-7 backbone LR.
- `k_sweep/`: active final K-schedule comparison at the selected 1e-6 LR.

Each config has a matching generated run directory under
`runs/stage2_cleanroom_v1/`.

## Current decision

E2 selected a common Phase-B backbone and added-parameter learning rate of
`1e-6`. The final K schedule and recurrent inference depth are not locked yet.
The eight-arm sweep is complete and is summarized in `results/k_sweep.md`.
It compares K=2, 90/10 K2/K3, 50/50 K2/K3, and K=3, all from the clean E1
wiring checkpoints with 1,048,576 unique tokens.

The historical mixed-K runs are development evidence only. The active mixed-K
arms sample K independently per
microbatch, record realized histograms and compute, and preserve scheduler/RNG
state in their checkpoints.
