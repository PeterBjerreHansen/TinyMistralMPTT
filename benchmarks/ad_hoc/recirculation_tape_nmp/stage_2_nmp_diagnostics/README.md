# Stage 2: NMP diagnostics and pressure sweep

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

## Gradient calibration and 1M pressure sweep

The short sweep uses a fresh, document-disjoint 5M artifact at
`data/dolmino/nmp_sweep_2048`. First run the calibration tool. It reports the
zero-output-head starting gradients, performs a head-only warm-up, and derives
fixed 5%, 10%, and 20% shared-backbone pressure weights:

```bash
uv run python benchmarks/ad_hoc/recirculation_tape_nmp/stage_2_nmp_diagnostics/calibrate_nmp.py \
  --config benchmarks/ad_hoc/recirculation_tape_nmp/stage_2_nmp_diagnostics/calibration.yaml
```

The resulting weights are frozen in the nine-arm study under
`short_sweep_1m/`. Run it with:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/ad_hoc/recirculation_tape_nmp/stage_2_nmp_diagnostics/short_sweep_1m \
  --skip-wire
```

The short sweep selects the pressure for the four final 5M continuations. It
logs NTP/NMP losses, shared-versus-head gradient norms, and NTP/NMP gradient
cosines; do not retune weights during a run.
