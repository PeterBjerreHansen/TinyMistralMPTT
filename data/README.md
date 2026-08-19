# Data artifacts

Dataset preparation recipes live beside the generated artifacts they define.
Only the small source recipes are tracked; token binaries, manifests, and other
materialized dataset files are ignored.

- `dolmino/local_2048/config.yaml`: small 2048-token development artifact.
- `dolmino/gpu_2048/config.yaml`: large 2048-token cloud/campaign artifact.

Use `scripts/prepare_data.py` and `scripts/verify_data.py` rather than editing a
generated artifact in place.
