# Wiring

Phase-A wiring runs establish MemoryAdd and MemoryTape32 starting points on the
active 2048-token substrate. `STUDY.yaml` records the purpose and runnable arms;
each arm's YAML is the authoritative execution configuration.

Generated checkpoints and logs are written to `results/generated/`. They are
local run artifacts and are intentionally not committed. Retain only compact
summaries in `results/` if a wiring result becomes scientifically useful.
