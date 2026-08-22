# Stage 3: auxiliary-objective runs

After reviewing Stage-2 diagnostics, run the NTP-only control and the serious
recurrent-only, tape-only, and dual-objective continuations from the exact same
Stage-1 hybrid parent checkpoint:

```bash
for objective in ntp recurrent tape dual; do
  uv run python scripts/train.py \
    --config benchmarks/ad_hoc/recirculation_tape_nmp/stage_3_nmp_training/serious_${objective}_2m.yaml \
    --resume-auto
done
```

Generated artifacts belong in this stage's `results/generated/` directory.

The `serious_ntp_2m` run has both NMP coefficients set to zero. It is a
same-checkpoint, same-data, same-pass-schedule control for attributing any
change in the auxiliary runs. The other runs keep NTP active and use the
default scale-aware coefficients: recurrent `0.01`, tape `0.10`, or both.
