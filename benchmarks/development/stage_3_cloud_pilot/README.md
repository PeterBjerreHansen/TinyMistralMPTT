# Stage 3: cloud pilot

After CUDA qualification, run every arm to the 5M gate:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_3_cloud_pilot \
  --skip-wire --until-unique-tokens 5242880
```

The configs declare 10M endpoints. Promoted seed-1337 arms resume to 10M by
rerunning selected arms without `--until-unique-tokens`.

The pilot benchmarks dense, periodic-C32, and write-only explicit-`<MEM>`-C32
Tape as separate arms. FBT is not part of this experimental pipeline.

Use the existing evaluation entry points on each 5M checkpoint:

```bash
uv run python scripts/evaluate_pass_depth.py --help
uv run python scripts/evaluate_recurrent_inference.py --help
uv run python scripts/evaluate_memory_interventions.py --help
```

Select promoted arms according to `../experimental_pipeline.md` before Stage 4.
