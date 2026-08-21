# Stage 1: local frozen-backbone wiring

Run all canonical wiring arms sequentially:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_1_wiring
```

Use `--arm <id>` to run one arm. Generated checkpoints and metrics live below
`results/generated/` and are ignored by Git. Tape is wired separately for
dense, periodic-C32, and write-only explicit-`<MEM>`-C32 writes.

These local configs keep only the newest checkpoint generation. Copy completed
wiring checkpoints to durable storage before removing local run directories.
