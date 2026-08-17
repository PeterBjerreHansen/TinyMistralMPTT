# Apply this update

This archive is an overlay for:

- repository: `PeterBjerreHansen/TinyMistralMPTT`
- branch: `mptt_v1`
- base commit: `c22e7c43c2d245d4b9338959b80230a7bfbd7a4e`

Copy the archive contents over the repository root. It intentionally contains
only files added or changed for K-general incremental/recurrent inference, so
unchanged files such as `uv.lock`, data, checkpoints, and run artifacts remain
untouched.

After applying, run:

```bash
uv run pytest -q
uv run python -m compileall -q src scripts tests
```

Then the first teacher-forced recurrent evaluation can use the already-trained
K=2 checkpoint, for example:

```bash
uv run python scripts/eval_recurrent.py \
  --config configs/mac/memory_add_phase_b_selected_lr1e-7_long.yaml \
  --checkpoint runs/mac-memory-add-phase-b-selected-lr1e-7-long/latest.pt \
  --prefill-passes 2 \
  --prompt-tokens 256 \
  --continuation-tokens 256
```
