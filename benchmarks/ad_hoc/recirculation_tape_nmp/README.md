# Recirculation–Tape NMP program

This ad-hoc program has three stages. It uses only the periodic-C32
Recirculation–Tape hybrid; dense and MemoryAdd variants are intentionally out of
scope for this pass. Each stage owns its execution code and its own
`results/generated/` directory; there is no shared results directory.

## Stage 1: 10M NTP continuation

Run the hybrid and vanilla configs on the document-disjoint pilot artifact:

```bash
uv run python scripts/train.py \
  --config benchmarks/ad_hoc/recirculation_tape_nmp/stage_1_ntp_continuation/hybrid_ntp_10m.yaml \
  --resume-auto

uv run python scripts/train.py \
  --config benchmarks/ad_hoc/recirculation_tape_nmp/stage_1_ntp_continuation/vanilla_ntp_10m.yaml \
  --resume-auto
```

The hybrid starts from the completed Stage-2 wired checkpoint. Vanilla starts
from the base TinyMistral checkpoint. These are useful reference trajectories,
but not a perfectly dose-matched comparison because the hybrid has already seen
the 5M wiring and 1M local-smoke training before this continuation.

The resulting hybrid checkpoint is the parent for every serious NMP run. Local
configs keep only the final checkpoint generation (`checkpoint_keep_last: 1`).

## Stage 2: objective-scale diagnostics

Run the diagnostic script on the wired checkpoint first, then repeat it against
the 10M hybrid checkpoint after Stage 1:

```bash
for scale in low default high; do
  uv run python benchmarks/ad_hoc/recirculation_tape_nmp/stage_2_nmp_diagnostics/diagnose_nmp.py \
    --config benchmarks/ad_hoc/recirculation_tape_nmp/stage_2_nmp_diagnostics/diagnostic_${scale}.yaml \
    --checkpoint benchmarks/development/stage_2_local_smoke/results/generated/\
hybrid_recirculation_smoke/checkpoints/checkpoint_000001048576.pt
done
```

The script evaluates the same eight training blocks at fixed K=2 without
updating weights. It reports NTP loss, RMS-normalized recurrent NMP loss, raw
post-writer sparse-tape NMP loss, target RMS, and the weighted auxiliary
contribution. The three files only vary the proposed coefficients (`low`,
`default`, `high`), so the objective comparison is not confounded by
optimization.

After 10M NTP, repeat the same commands without `--checkpoint`; each diagnostic
config then uses the declared 10M parent automatically. The first acceptance
check is that NTP, recurrent NMP, and tape NMP are within a sensible order of
magnitude after accounting for target RMS and the configured coefficients.

## Stage 3: serious auxiliary runs

Once the diagnostic JSON is reviewed, run the NTP-only control and the three
2M-token auxiliary continuations:

```bash
for objective in ntp recurrent tape dual; do
  uv run python scripts/train.py \
    --config benchmarks/ad_hoc/recirculation_tape_nmp/stage_3_nmp_training/serious_${objective}_2m.yaml \
    --resume-auto
done
```

The NTP control has both NMP coefficients set to zero and uses the exact same
starting checkpoint, data, pass schedule, and token budget as the auxiliary
runs. The other runs keep NTP active and use the default scale-aware
coefficients: recurrent `0.01`, tape `0.10`, or both.
The 2M continuation reuses the pilot training artifact because no separate
100M local artifact is materialized. Validation remains the pilot validation
split, which is separate from its training split. Treat these as continuation
experiments, not fresh-data generalization estimates.

The intended comparison is whether sparse tape NMP improves long-horizon
prediction without harming NTP, and whether adding recurrent NMP provides an
additional benefit. The recurrent target is RMS-normalized by default; use the
raw-state mode only as an explicit ablation.
