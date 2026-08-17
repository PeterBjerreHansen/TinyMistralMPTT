# Stage 2 locked configurations

This directory is intentionally empty of model-training YAMLs while the Stage 2
training protocol is still under development.

Development configs live under `experiments/stage2_training/protocol_development/`.
After pass depth, loss weighting, data scale, and other protocol choices are
locked, the three principal configs will be promoted here:

- `mac/vanilla.yaml`
- `mac/memory_add.yaml`
- `mac/memory_tape32.yaml`

The main-run configs must start from the canonical Stage 1 starting points in
`experiments/stage1_starting_points/STARTING_POINTS.yaml`. A development
checkpoint is never promoted implicitly just because it is the newest model.
