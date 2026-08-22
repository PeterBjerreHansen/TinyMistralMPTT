# Stage 2: NMP diagnostics

Run the three coefficient diagnostics against the wired checkpoint first and
then against the completed Stage-1 hybrid checkpoint:

```bash
for scale in low default high; do
  uv run python benchmarks/ad_hoc/recirculation_tape_nmp/stage_2_nmp_diagnostics/diagnose_nmp.py \
    --config benchmarks/ad_hoc/recirculation_tape_nmp/stage_2_nmp_diagnostics/diagnostic_${scale}.yaml \
    --checkpoint benchmarks/development/stage_2_local_smoke/results/generated/hybrid_recirculation_smoke/checkpoints/checkpoint_000001048576.pt
done
```

Reports belong in this stage's `results/generated/` directory. Re-run without
`--checkpoint` after Stage 1 completes so the configs use the 10M parent.
