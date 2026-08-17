# FBT retrofit experiments

This directory preserves one-off FBT adaptation experiments without making
those knobs part of the stable training API.

The stable implementation remains `tiny_mistral_mptt.variants.FBTVariant`:
pass 1 is exact TinyMistral and later passes use the asymmetric GLU feedback
operator. `prefix_mixin_probability` is still supported because prefix mixing
changes the training trajectory and was used by a recorded experiment.

Historical configs:

- `configs/fbt_adaptation.yaml`: prefix-mixed Phase-B pilot;
- `configs/fbt_coadaptation.yaml`: prefix-free Phase-B co-adaptation pilot;
- `configs/fbt_calibrated_init.historical.yaml`: calibrated-initialization run.

The calibrated-initialization hook was retired from `ExperimentConfig` and
`scripts/train.py`. It was useful diagnostically but did not become a mainline
training method. `calibration_reference.py` preserves the exact rescaling
procedure for reproducibility without importing it into the package.

`diagnose_fbt.py` is also experiment-local because its metrics are specific to
this retrofit investigation.
