# Stage 1: continued NTP

This stage continues the wired Recirculation–Tape hybrid and trains a vanilla
reference on the 10M-token pilot artifact. The hybrid is the parent checkpoint
for the later NMP diagnostics and auxiliary-objective runs.

```bash
uv run python scripts/train.py \
  --config benchmarks/ad_hoc/recirculation_tape_nmp/stage_1_ntp_continuation/hybrid_ntp_10m.yaml \
  --resume-auto

uv run python scripts/train.py \
  --config benchmarks/ad_hoc/recirculation_tape_nmp/stage_1_ntp_continuation/vanilla_ntp_10m.yaml \
  --resume-auto
```

Generated artifacts belong in this stage's `results/generated/` directory.
Keep only the final local checkpoint generation.

The completed hybrid run is retained at
`results/generated/hybrid_ntp_10m/`. The temporary shared ad-hoc results tree
used while that run was active has been removed.
