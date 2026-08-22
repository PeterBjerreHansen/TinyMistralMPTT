# Stage 3: auxiliary-objective runs

After reviewing Stage-2 diagnostics, run the NTP-only control and the serious
recurrent-only, tape-only, and dual-objective continuations from the exact same
Stage-1 hybrid parent checkpoint:

```bash
for objective in ntp recurrent tape dual; do
  uv run python scripts/train.py \
    --config benchmarks/ad_hoc/recirculation_tape_nmp/stage_3_nmp_training/serious_${objective}_5m.yaml \
    --resume-auto
done
```

Generated artifacts belong in this stage's `results/generated/` directory.

All four arms consume the prepared 5M-token `data/dolmino/nmp_sweep_2048`
artifact once and use a constant learning rate. The earlier cosine-decay
Stage-1 trajectory remains available separately.

The `serious_ntp_5m` run has both NMP coefficients set to zero. It is a
same-checkpoint, same-data, same-pass-schedule control for attributing any
change in the auxiliary runs. The recurrent and tape arms use the calibrated
20% objective weights (`14.412281` and `3.404028`, respectively). The dual arm
uses half of each coefficient (`7.2061405` and `1.702014`) so that its total
auxiliary pressure is comparable to the single-objective arms.
