# Memory wiring and Phase-B campaign

This directory is the reproducibility record for the completed MemoryAdd and
MemoryTape32 frozen-wiring / Phase-B dose-response campaign. Keeping the sweep
configs here prevents the stable `configs/mac/` directory from becoming an
experiment ledger.

- `WIRED_BENCHMARK.md` records the mature 1,048,576-token frozen-backbone wiring
  checkpoints and interventions.
- `DOSE_RESPONSE.md` records the backbone-LR pilot and the one-million-token
  Phase-B continuations.
- `configs/memory_add/` and `configs/memory_tape32/` preserve the exact configs
  used for those runs, including zero-LR controls and selected continuations.

The current mainline operating point distilled from this campaign is encoded in
`configs/mac/memory_add_phase_b.yaml` and
`configs/mac/memory_tape32_phase_b.yaml`: K=2 training, pass weights
`[0.25, 0.75]`, pretrained LR `1e-7`, added LR `1e-6`, constant LR schedule.
The stable Phase-B configs deliberately leave `init_from: null`; pass the mature
wired checkpoint explicitly with `--init-from` when starting a new Phase-B run.
