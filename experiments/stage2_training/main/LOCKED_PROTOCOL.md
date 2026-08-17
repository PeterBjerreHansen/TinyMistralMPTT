# Stage 2 protocol lock

**Status: NOT LOCKED**

No MemoryAdd or MemoryTape32 main-run config should be placed in
`configs/stage2/` until this status changes.

## Already supported by development evidence

- canonical starting points: frozen-wired MemoryAdd and MemoryTape32 artifacts
  in `experiments/stage1_starting_points/STARTING_POINTS.yaml`;
- conservative pretrained LR: `1e-7`;
- added-parameter LR: `1e-6`;
- fixed-K training and K-general evaluation/inference are both operational;
- exact/recurrent cached inference is healthy for the two selected memory
  models.

## Still to lock

At minimum:

- training pass depth K and associated loss weights;
- main training token/compute budget;
- fresh larger training dataset and held-out split policy;
- checkpoint/evaluation milestones;
- vanilla compute-matching policy.

## Promotion rule

When locked, record the complete protocol here, change the status to `LOCKED`,
and create exactly the principal model configs under `configs/stage2/`.
Those configs should use `init_from` from the Stage 1 manifest (or the base
TinyMistral control), never an implicit development descendant.
