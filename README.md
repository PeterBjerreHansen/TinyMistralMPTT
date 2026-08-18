# TinyMistralMPTT

Research code for MemoryAdd, MemoryTape32, and recurrent inference on a
validated TinyMistral backbone.

## Read first

- `configs/`: canonical runnable configs.
- `experiments/stage2_cleanroom_v1/`: complete protocol lineage, controls, and
  results.
- `runs/`: generated checkpoints and metrics; ignored by Git.
- `docs/`: durable implementation and provenance contracts.
- `src/tiny_mistral/`: validated vendored vanilla implementation.
- `src/tiny_mistral_mptt/`: research models, training, evaluation, and inference.

The checkpoint directory is `checkpoints/TinyMistral-248M-v3/`. This exact name
is retained because it is the upstream model identifier, not this repository's
version. The upstream revision is pinned in `docs/UPSTREAMS.md`.

## Locked Stage 2 protocol

The main configs are:

```text
configs/stage2/memory_add_k3.yaml
configs/stage2/memory_tape32_k3.yaml
```

They use K=3, pass weights `[0.05, 0.20, 0.75]`, backbone learning rate `3e-7`,
added-parameter learning rate `1e-6`, and 1,048,576 unique training tokens.
K=2 and compute-matched K=3 remain controls in the clean-room lineage.

## Validate

```bash
uv sync --extra data --extra eval
uv run pytest -q
uv run python -m compileall -q src scripts tests experiments
git diff --check
```

Prepare and verify the clean-room data artifact:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/verify_data.py data/stage2_cleanroom_v1/sequence_512
```

Train a main model:

```bash
uv run python scripts/train.py --config configs/stage2/memory_add_k3.yaml
```

Evaluate pass depth, memory use, or recurrent inference with the scripts in
`scripts/` and the selected checkpoint under `runs/stage2/`.
