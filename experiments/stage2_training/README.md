# Stage 2 — training and comparison

Stage 2 contains both protocol development and the eventual locked main run.
This is intentional: LR selection, K selection, loss weighting, and recurrence
checks are training experiments that inform the final protocol.

- `protocol_development/` contains targeted ablations and health experiments.
- `main/` contains the lock decision and, later, the principal comparison
  record.

Unless a development experiment explicitly studies checkpoint ancestry, a
clean comparison should start from the selected Stage 1 artifacts in
`../stage1_starting_points/STARTING_POINTS.yaml`.
