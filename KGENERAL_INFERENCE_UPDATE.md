# K-general incremental/recurrent inference update

Base branch: `mptt_v1` at `c22e7c43c2d245d4b9338959b80230a7bfbd7a4e`.

This update adds explicit cached inference for the successfully trained
`memory_add` and `memory_tape32` variants. It does not alter the validated
`src/tiny_mistral/` backbone and does not change the existing public
`MultiPassVariant.generate()` semantics.

## Added inference contract

`prefill_passes=K` is an inference-time hyperparameter for any positive K.

- `exact_incremental`: K independent TinyMistral KV streams. Stream `k>1`
  consumes strict-past feedback from stream `k-1`. This is an incremental
  oracle for finite K-pass recomputation and costs K cached backbone steps per
  continuation token.
- `recurrent`: K-pass prompt prefill followed by one continuing final-pass
  stream. For K>1 the state is seeded with the pass-K KV cache and pass-(K-1)
  feedback memory; after the first appended token the newly produced pass-K
  state closes the recurrent loop. K=1 is ordinary vanilla cached inference.

All exact-stream source memories are snapshotted before the token is processed,
so a newly produced same-position state cannot leak into a higher pass.

## Architecture-specific state

MemoryAdd retains exactly one previous top-layer vector. Its incremental path
uses that vector directly and therefore does not apply the full-sequence
right-shift a second time.

MemoryTape32 retains an oldest-to-newest ring of at most `memory_window`
previous top-layer vectors. Cached one-token memory reads use the new
`memory_bank_attention` primitive, because every supplied bank element is
already strictly in the past.

## Evaluation

`src/tiny_mistral_mptt/evaluation/recurrent.py` and
`scripts/eval_recurrent.py` perform teacher-forced held-out continuation
scoring. For one or more K values they report:

- exact incremental K NLL;
- collapsed recurrent K NLL;
- pass-1/vanilla cached NLL from the same checkpoint;
- recurrent-minus-exact and recurrent-minus-vanilla gaps by horizon;
- per-offset NLL;
- recurrent/exact final-hidden RMS distance and cosine similarity.

Example:

```bash
uv run python scripts/eval_recurrent.py \
  --config configs/mac/memory_add_phase_b_selected_lr1e-7_long.yaml \
  --checkpoint runs/mac-memory-add-phase-b-selected-lr1e-7-long/latest.pt \
  --prefill-passes 1 2 3 4 8 \
  --prompt-tokens 256 \
  --continuation-tokens 256 \
  --horizons 1 2 4 8 16 32 64 128 256
```

The first scientific run should still use K=2 because that is the trained
mainline regime. Other K values are inference-depth diagnostics until models
are explicitly trained at those depths.

## Validation in this build

- full CPU/reference suite: `126 passed, 8 skipped`;
- the 8 skips are MPS-only tests in this Linux environment;
- exact incremental versus full recomputation is tested for K=1,2,3,4 for both
  MemoryAdd and MemoryTape32, including sequences longer than the TinyMistral
  sliding-attention window;
- recurrent handoff parity is tested for K=2,3,4;
- K=1 is tested against ordinary vanilla cached inference;
- MemoryTape32 ring ordering/bounds and MemoryAdd one-vector state are tested;
- MPS smoke coverage includes K-general exact and recurrent paths and will run
  on Apple hardware;
- all 26 Mac experiment configs parse;
- `python -m compileall -q src scripts tests` passes;
- the vanilla source manifest test passes, confirming `src/tiny_mistral/` is
  unchanged.

The GitHub integration available in this environment is read-only for this
repository, so these source changes are distributed as an archive rather than
pushed to `mptt_v1`.
