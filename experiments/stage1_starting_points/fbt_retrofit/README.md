# FBT retrofit development

## Question

Can the exact FBT-style one-state fusion be retrofitted into the validated
pretrained TinyMistral strongly enough to serve as a fair Stage 1 starting
point?

## Record

The stable implementation remains `tiny_mistral_mptt.variants.FBTVariant`.
This directory preserves the Phase-A/Phase-B configs plus prefix-mixing,
co-adaptation, and calibrated-initialization investigations.

- `configs/fbt_phase_a.yaml` / `fbt_phase_b.yaml`: basic retrofit trajectory;
- `configs/fbt_adaptation.yaml`: prefix-mixed adaptation pilot;
- `configs/fbt_coadaptation.yaml`: prefix-free co-adaptation pilot;
- `configs/fbt_calibrated_init.historical.yaml`: historical calibrated-init run.

The calibrated-initialization hook was intentionally retired from the stable
config/model API; `calibration_reference.py` preserves the procedure without
reintroducing one-off machinery. `diagnose_fbt.py` is experiment-local for the
same reason.

## Decision

The exact retrofit did not produce a satisfactory canonical Stage 1 starting
point. FBT remains an informative comparison and implementation reference, but
MemoryAdd is the selected one-state recurrent stand-in for the main training
comparison.
