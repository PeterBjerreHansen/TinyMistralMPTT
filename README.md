# TinyMistralMPTT

Research code for retrofitting multi-pass latent-state mechanisms onto the
validated `M4-ai/TinyMistral-248M-v3` backbone.

The repository keeps the validated TinyMistral implementation under
`src/tiny_mistral/` and places research behavior under
`src/tiny_mistral_mptt/`. The current comparison has four first-class variants:

- `vanilla`: exact one-pass TinyMistral control;
- `fbt`: asymmetric-GLU feedback from the previous pass's immediately preceding
  top-layer state;
- `memory_add`: a zero-initialized additive residual from one preceding
  top-layer state;
- `memory_tape32`: per-layer GQA cross-attention to the previous pass's recent
  strictly earlier top-layer states.

No chunked-memory or hybrid architecture is implemented in this phase.

## Scientific contracts

For every multipass variant, **pass 1 is the vanilla model**. Research modules
are bypassed entirely rather than merely gated to zero. MemoryAdd is additionally
zero-initialized, so every pass depth is exactly vanilla before training.

Training is K-general. Pass count is a configurable hyperparameter sampled from
a token-indexed schedule, while per-pass losses use independently configurable,
right-aligned non-negative weights. Phase A freezes the pretrained backbone;
Phase B unfreezes it and uses independent learning rates for pretrained and
architecture-added parameters.

`resume_from` restores an exact optimizer/data/RNG/pass-scheduler trajectory.
`init_from` loads model weights only and starts a fresh training trajectory.
See `docs/EXPERIMENT.md` for the complete contract.

## Install and validate

```bash
uv sync --extra data --extra eval
uv run pytest -q
```

Download and verify the pinned TinyMistral checkpoint:

```bash
uv run python scripts/download_checkpoint.py
uv run python scripts/verify_checkpoint.py
uv run python scripts/compare_hf.py --device cpu --dtype float32
uv run python scripts/compare_hf_layers.py --device cpu --dtype float32
uv run python scripts/compare_hf_inputs_embeds.py --length 40
```

The vendored backbone is provenance-checked by `docs/VANILLA_SOURCE.sha256`.
`docs/UPSTREAMS.md` records the exact source/checkpoint revisions and the single
non-numerical dispatch hardening relative to the validated TinyMistralFork
commit.

## Development data

```bash
uv run python scripts/prepare_data.py \
  --config configs/data/dolmino_dev_512.yaml
uv run python scripts/verify_data.py data/dolmino/dev_512
```

The development recipe materializes 1,048,576 training tokens and 131,072 held-
out tokens at sequence length 512 from a pinned Dolmino mixture. Generated data
is local and ignored by git.

## Core training workflow

Vanilla control:

```bash
uv run python scripts/train.py --config configs/mac/vanilla.yaml
```

Frozen wiring:

```bash
uv run python scripts/train.py --config configs/mac/memory_add_phase_a.yaml
uv run python scripts/train.py --config configs/mac/memory_tape32_phase_a.yaml
```

The canonical memory Phase-A configs run to 1,048,576 frozen-backbone tokens,
matching the mature wiring stage used in the controlled comparison.

Start Phase B from an explicitly preserved wired checkpoint:

```bash
uv run python scripts/train.py \
  --config configs/mac/memory_add_phase_b.yaml \
  --init-from checkpoints/memory_add_frozen_wired_v1.pt

uv run python scripts/train.py \
  --config configs/mac/memory_tape32_phase_b.yaml \
  --init-from checkpoints/memory_tape32_frozen_wired_v1.pt
```

The mainline Phase-B operating point is K=2, loss weights `[0.25, 0.75]`,
pretrained LR `1e-7`, added LR `1e-6`, constant schedule. Pass depth remains a
hyperparameter: this operating point is an experimental choice, not an API
restriction.

FBT remains available as a comparison:

```bash
uv run python scripts/train.py --config configs/mac/fbt_phase_a.yaml
uv run python scripts/train.py \
  --config configs/mac/fbt_phase_b.yaml \
  --init-from runs/mac-fbt-phase-a/latest.pt
```

One-off FBT retrofit experiments are archived under `experiments/fbt_retrofit/`
rather than exposed as stable training knobs.

## Finite-pass evaluation

```bash
uv run python scripts/eval_pass_depth.py \
  --config configs/mac/memory_add_phase_b.yaml \
  --checkpoint <checkpoint.pt> \
  --passes 8

uv run python scripts/eval_memory_interventions.py \
  --config configs/mac/memory_add_phase_b.yaml \
  --checkpoint <checkpoint.pt>
```

Pass-depth evaluation reports NLL/perplexity at every requested depth, per-source
NLL, and hidden-state delta RMS. Memory interventions compare real, zeroed, and
mismatched previous state; for MemoryAdd they also report recurrent-residual RMS
relative to token-embedding RMS.

## Cached exact and recurrent inference

MemoryAdd and MemoryTape32 implement explicit K-general cached inference under
`tiny_mistral_mptt.inference`:

- `exact_incremental`: K independent TinyMistral KV streams, exactly matching
  finite K-pass recomputation;
- `recurrent`: the same K-pass prompt prefill, followed by one continuing final-
  pass KV stream with recurrent feedback.

K=1 is an exact vanilla boundary case. For K>1, recurrent mode is seeded from
pass K-1, so the first processed continuation token matches exact K-pass
inference; approximation begins only after the feedback loop closes.

Teacher-forced comparison:

```bash
uv run python scripts/eval_recurrent.py \
  --config configs/mac/memory_add_phase_b.yaml \
  --checkpoint <checkpoint.pt> \
  --prefill-passes 1 2 3 4 \
  --prompt-tokens 256 \
  --continuation-tokens 256
```

Public multipass `generate()` intentionally retains vanilla semantics. Recurrent
behavior is opt-in through the explicit inference API. FBT does not currently
implement cached feedback inference beyond the K=1 vanilla boundary.

See `docs/RECURRENT_INFERENCE.md` for state, cache, and causality contracts.

## Experiment history

Completed campaign artifacts and active continuation provenance are separated from
the mature baseline surface:

- `experiments/memory_phase_b/`: mature frozen wiring, LR dose response, matched
  controls, completed Phase-B continuations, active K=3 source configs, and their
  exact configs;
- `experiments/fbt_retrofit/`: prefix-mixing/co-adaptation/calibrated-init
  investigations and FBT-specific diagnostics.

This preserves reproducibility without turning `configs/mac/`, `scripts/`, or
`src/` into an experiment ledger.

## Repository layout

```text
src/tiny_mistral/                   validated vendored TinyMistral backbone
src/tiny_mistral_mptt/attention/    research-only local memory attention
src/tiny_mistral_mptt/data/         Dolmino materialization and packed dataset
src/tiny_mistral_mptt/training/     phases, pass/LR schedules, checkpointing
src/tiny_mistral_mptt/variants/     vanilla, FBT, MemoryAdd, MemoryTape32
src/tiny_mistral_mptt/evaluation/   NLL, pass-depth, recurrent, lm-eval adapter
src/tiny_mistral_mptt/inference/    K-general exact/recurrent cached inference
configs/                            current data/Mac/GPU/smoke configs
experiments/                        completed campaign records and sweep configs
eval_configs/                       external benchmark suites
docs/                               protocol, provenance, validation contracts
tests/                              offline numerical/causality/trajectory tests
```

See `docs/VALIDATION.md` for the validated gates and
`experiments/memory_phase_b/` for the main memory results obtained so far.
