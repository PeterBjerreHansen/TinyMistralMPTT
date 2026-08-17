# Experiment records

`experiments/` answers scientific questions; it does not define reusable model
APIs.

The project uses this lifecycle:

```text
substrate validation
    ↓
Stage 1 — starting points
    ↓
Stage 2 — protocol development → locked main comparison
    ↓
capability ablations
```

## Record convention

Each development experiment should be understandable from a small README with
these headings when applicable:

1. **Question**
2. **Starting checkpoints**
3. **Protocol**
4. **Measurements**
5. **Results**
6. **Decision**

Exact YAMLs used for completed runs belong beside that record. Generic scripts
stay in `scripts/`; reusable implementation stays in `src/`.

## Promotion rule

Development configs do not become canonical because they are recent or because
they produced the best checkpoint so far. Promotion requires an explicit
scientific decision:

- Stage 1 selected checkpoints are recorded in
  `stage1_starting_points/STARTING_POINTS.yaml`.
- Stage 2 main configs are promoted into `configs/stage2/` only after
  `stage2_training/main/LOCKED_PROTOCOL.md` is marked locked.
