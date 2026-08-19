# Substrate controls

Vanilla TinyMistral reference runs for the local Mac and GPU environments.

- `mac.yaml`: local 2048-token control.
- `gpu.yaml`: long GPU preflight/control configuration. It deliberately starts at `batch_size=1`, `grad_accum_steps=1` so CUDA capacity does not silently change the 2,048-token optimizer batch validated in development.

Raw outputs are written under `results/generated/` and ignored. Retain only
compact comparison notes when a substrate result matters scientifically.
