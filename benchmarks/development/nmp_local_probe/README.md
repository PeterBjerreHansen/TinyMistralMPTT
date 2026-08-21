# Local NMP probe

This is a short paired experiment, not an architecture ranking. Each pair
starts from the same completed Stage-2 NTP checkpoint, uses the document-
disjoint pilot dataset, consumes the same 131,072 linguistic tokens, and differs
only in its NMP objective and ramp.

The initial weights account for the different raw latent-loss scales observed
during full-model wiring:

- MemoryAdd recurrent NMP: `0.05`;
- periodic-C32 Tape NMP: `0.10`;
- Recirculation–Tape dual NMP: recurrent `0.01`, tape `0.10`.

All NMP weights ramp from zero over 32,768 tokens. NTP stays enabled with the
same pass weights in every arm. These runs answer whether training is stable,
whether the prediction losses start decreasing, and whether any validation-NLL
signal is already visible. They are too short to establish a model-quality
claim.

Run all pairs sequentially:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/nmp_local_probe
```

Run or resume one arm with `--arm ARM_ID`. Generated checkpoints and metrics
live below `results/generated/` and are ignored by Git.
