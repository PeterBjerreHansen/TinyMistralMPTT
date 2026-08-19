# K selection

This completed development qualification compares fixed K=2 and fixed K=3
within MemoryAdd and MemoryTape32 on the active 2048-token substrate.
`STUDY.yaml` declares the two within-architecture comparisons. The runnable
configs contain the actual token budgets, learning rates, pass schedules, and
loss weights; those values are intentionally not duplicated in the manifest.

The retained short qualification is `results/baseline_2048.md`. Generated
checkpoints and logs belong under `results/generated/` and are ignored.
