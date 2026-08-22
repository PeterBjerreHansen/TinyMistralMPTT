# Stage 1: continued NTP

This stage continues the wired Recirculation–Tape hybrid and trains a vanilla
reference on the 10M-token pilot artifact. The hybrid is the parent checkpoint
for the later NMP diagnostics and auxiliary-objective runs.

```bash
uv run python scripts/train.py \
  --config benchmarks/ad_hoc/recirculation_tape_nmp/stage_1_ntp_continuation/hybrid_ntp_10m.yaml && \
uv run python scripts/train.py \
  --config benchmarks/ad_hoc/recirculation_tape_nmp/stage_1_ntp_continuation/vanilla_ntp_10m.yaml
```

Generated artifacts belong in this stage's `results/generated/` directory.
Keep only the final local checkpoint generation.

The earlier cosine-decay run is retained at `results/generated/hybrid_ntp_10m/`.
The no-decay rerun writes to `results/generated/hybrid_ntp_10m_constant_lr/`
and the vanilla rerun writes to `results/generated/vanilla_ntp_10m_constant_lr/`.
