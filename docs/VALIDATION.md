# Bootstrap validation

This document records what was executable in the artifact-building environment
and what remains an external integration gate.

## Executed offline

- Python source compilation for `src/`, `scripts/`, and `tests/`.
- Full offline unit/contract suite: 62 passed, 1 skipped.
- The skipped test is the real Apple MPS hardware smoke test; the build host is
  Linux/CPU.
- TinyMistral component, masking, cache, generation, state-dict, local-attention,
  and training tests inherited from the vanilla reference substrate.
- Vanilla experiment wrapper identity and Phase-A/Phase-B trainability contract.
- Deterministic Dolmino recipe allocation and synthetic document materialization.
- Fixed `uint16` mmap artifact reads and source labels.
- Stateful shuffled block-sampler exact restore.
- Experiment checkpoint model/optimizer/sampler/counter restore and rejection of
  trajectory-changing config changes.
- End-to-end tiny vanilla trainer, held-out NLL, exact token-budget termination,
  final partial gradient accumulation, and bit-exact parameter agreement between
  interrupted/resumed and uninterrupted CPU training.
- Optional lm-eval module imports cleanly when the optional dependency is absent.
- Editable local package install/import with build isolation disabled (the build
  host has no package-index network access).

## External gates not executable here

The build environment has no general outbound package/model/dataset access and
no MPS/CUDA accelerator. Before treating a local experiment as validated, run:

```bash
uv sync --extra data --extra eval
uv run pytest -q
uv run python scripts/download_checkpoint.py
uv run python scripts/verify_checkpoint.py
uv run python scripts/compare_hf.py --device cpu --dtype float32
uv run python scripts/compare_hf_layers.py --device cpu --dtype float32
uv run python scripts/compare_hf_inputs_embeds.py --length 40
uv run python scripts/mps_smoke.py
uv run python scripts/prepare_data.py --config configs/data/dolmino_dev_512.yaml
uv run python scripts/train.py --config configs/mac/vanilla.yaml
uv run python scripts/eval_nll.py --config configs/mac/vanilla.yaml --max-blocks 32
uv run python scripts/eval_lm.py --config configs/mac/vanilla.yaml --suite eval_configs/quick.yaml --limit 10
```

The last command is deliberately a small adapter/integration check. Increase the
limit only after it succeeds.
