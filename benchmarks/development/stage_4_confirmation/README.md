# Stage 4: selected confirmation

First resume each promoted Stage-3 seed to its declared 10M endpoint:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_3_cloud_pilot \
  --skip-wire --arm vanilla_seed1337 --arm tape_seed1337 \
  --arm hybrid_seed1337 --arm <selected-fast-seed1337>
```

Then materialize the two additional-seed study. Choose the fast baseline using
the Stage-3 gate:

```bash
uv run python benchmarks/development/stage_4_confirmation/prepare.py \
  --fast memory_add
uv run python scripts/verify_study.py \
  benchmarks/development/stage_4_confirmation
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_4_confirmation --skip-wire
```

`--fast` accepts `memory_add`, `fbt`, or `recirculation_adaptive`. Preparation
fails rather than overwriting an existing selection; pass `--force` only when
deliberately replacing a study that has not begun execution.

