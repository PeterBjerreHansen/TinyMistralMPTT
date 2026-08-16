# TinyMistralMPTT

A self-contained research repository for retrofitting multi-pass latent-state
mechanisms onto the validated `M4-ai/TinyMistral-248M-v3` backbone.

The repository deliberately keeps `src/tiny_mistral/` as the vanilla reference
implementation and places all research behavior under `src/tiny_mistral_mptt/`.
The current comparison supports four first-class variants:

- `vanilla`: exact one-pass TinyMistral control;
- `fbt`: asymmetric-GLU latent feedback from the previous pass's immediately
  preceding top-layer state;
- `memory_add`: a zero-initialized additive residual from the previous pass's
  immediately preceding top-layer state;
- `memory_tape32`: per-layer GQA cross-attention to the previous pass's most
  recent 32 strictly earlier top-layer states.

No chunked-memory or hybrid model is implemented in this phase.

## Scientific invariants

For every multipass variant, **pass 1 is the vanilla model**. Newly initialized
research modules are skipped entirely on pass 1. A one-pass FBT, MemoryAdd, or
MemoryTape32 forward must therefore reproduce vanilla TinyMistral exactly.
MemoryAdd is additionally zero-initialized so *all* pass depths reproduce vanilla
exactly before training.

Training is intentionally configurable rather than tied to one paper-specific
recipe:

- pass counts are sampled from a token-indexed configurable schedule;
- pass losses use configurable, right-aligned non-negative weights;
- Phase A freezes the pretrained backbone and trains only added modules;
- Phase B has independent base learning rates for pretrained and added
  parameters, so the backbone may adapt slowly rather than being all-or-nothing
  frozen;
- LR multipliers may be constant, cosine, or piecewise-linear;
- `init_from` loads model weights only for a fresh experimental stage;
- `resume_from` restores the exact optimizer/data/RNG/pass-scheduler trajectory.

See `docs/EXPERIMENT.md` for the full protocol.

## 1. Install and test

```bash
uv sync --extra data --extra eval
uv run pytest -q
```

Python 3.13 is selected by `.python-version`. `uv.lock` is committed by the
online development environment.

## 2. Download and verify TinyMistral

```bash
uv run python scripts/download_checkpoint.py
uv run python scripts/verify_checkpoint.py
```

Strong vanilla oracle checks remain available:

```bash
uv run python scripts/compare_hf.py --device cpu --dtype float32
uv run python scripts/compare_hf_layers.py --device cpu --dtype float32
uv run python scripts/compare_hf_inputs_embeds.py --length 40
uv run python scripts/mps_smoke.py
```

## 3. Materialize the Dolmino development artifact

```bash
uv run python scripts/prepare_data.py \
  --config configs/data/dolmino_dev_512.yaml
uv run python scripts/verify_data.py data/dolmino/dev_512
```

The checked-in development config produces 1,048,576 training tokens and
131,072 held-out tokens at sequence length 512 from the pinned published
Dolmino 50B Stage-2 mixture. Data files are local artifacts and are not checked
into git.

## 4. Vanilla control

```bash
uv run python scripts/train.py --config configs/mac/vanilla.yaml
uv run python scripts/eval_nll.py --config configs/mac/vanilla.yaml
uv run python scripts/eval_nll.py \
  --config configs/mac/vanilla.yaml \
  --checkpoint runs/mac-vanilla/latest.pt
```

The vanilla model has no Phase A.

## 5. FBT wiring and adaptation

Phase A trains only the two added feedback projections with fixed two-pass
training. The checked-in config is a **starting point, not a canonical
objective**:

```bash
uv run python scripts/train.py --config configs/mac/fbt_phase_a.yaml
```

Phase B initializes from the Phase-A model weights, unfreezes the backbone, and
uses separate pretrained/added learning rates:

```bash
uv run python scripts/train.py --config configs/mac/fbt_phase_b.yaml
```

Evaluate the recurrent pass map explicitly:

```bash
uv run python scripts/eval_pass_depth.py \
  --config configs/mac/fbt_phase_b.yaml \
  --checkpoint runs/mac-fbt-phase-b/latest.pt \
  --passes 8
```

## 6. MemoryAdd one-state control

MemoryAdd keeps the current token embedding on the normal pretrained path and
adds a residual derived from the previous pass's immediately preceding top-layer
state:

```text
x_t = e_t + W_M RMSNorm(h_(t-1)^(k-1))
```

`W_M` is zero-initialized. Consequently pass 2 (and deeper passes) are exact
vanilla before training, making this a clean test of whether a single recurrent
latent can learn a useful correction without replacing TinyMistral's input
representation. Phase A freezes the backbone and trains only the added RMSNorm
and projection.

```bash
uv run python scripts/train.py --config configs/mac/memory_add_phase_a.yaml

uv run python scripts/eval_pass_depth.py \
  --config configs/mac/memory_add_phase_a.yaml \
  --checkpoint runs/mac-memory-add-phase-a/latest.pt \
  --passes 8

uv run python scripts/eval_memory_interventions.py \
  --config configs/mac/memory_add_phase_a.yaml \
  --checkpoint runs/mac-memory-add-phase-a/latest.pt
```

The intervention evaluator compares real, zeroed, and mismatched previous states
and reports the MemoryAdd residual/embedding RMS ratio. Phase B is checked in as
a conservative next-stage config but should only be run after the Phase-A depth
and intervention gates are inspected.

## 7. MemoryTape32 wiring and adaptation

MemoryTape32 is an equal-status research variant, not scaffolding for another
model. Each decoder layer reads the previous pass's top-layer memory tape via a
strict-past local GQA reader with default window 32.

```bash
uv run python scripts/train.py \
  --config configs/mac/memory_tape32_phase_a.yaml

uv run python scripts/train.py \
  --config configs/mac/memory_tape32_phase_b.yaml
```

Pass-depth evaluation:

```bash
uv run python scripts/eval_pass_depth.py \
  --config configs/mac/memory_tape32_phase_b.yaml \
  --checkpoint runs/mac-memory-tape32-phase-b/latest.pt \
  --passes 8
```

## 8. Flexible objectives and schedules

Pass supervision is independent of pass-count sampling. For example:

```yaml
pass_schedule:
  - until_tokens: 2000000
    probabilities:
      2: 1.0
  - probabilities:
      1: 0.50
      2: 0.45
      3: 0.05

pass_loss_weights: [0.05, 0.20, 0.75]
```

If a two-pass batch is sampled, the final two configured weights are used and
renormalized. A one-pass batch always reduces to ordinary one-pass NTP.

Phase-B parameter groups are independent:

```yaml
pretrained_learning_rate: 1.0e-7
added_learning_rate: 1.0e-6
lr_schedule:
  type: cosine
  warmup_tokens: 16384
  min_multiplier: 0.1
```

A manual schedule is also supported:

```yaml
lr_schedule:
  type: piecewise_linear
  points:
    - [0,        0.2]
    - [100000,   1.0]
    - [5000000,  1.0]
    - [20000000, 0.1]
```

The multiplier is applied to both parameter groups while preserving their LR
ratio.

## 9. `init_from` versus `resume_from`

`resume_from` means exact continuation. Model, optimizer, data sampler, Python
and PyTorch RNG, pass-scheduler RNG/histogram, counters, and phase are restored.
Trajectory-changing config edits are rejected.

`init_from` loads only model parameters and starts a fresh run with new
optimizer/scheduler/data counters. This is the intended Phase-A -> Phase-B
transition.

## 10. Evaluation boundary

`eval_nll.py` and the existing `lm-evaluation-harness` adapter retain **one-pass
vanilla semantics** by default. This is deliberate: ordinary model calls still
mean the standard TinyMistral model.

`eval_pass_depth.py` is the current multipass research evaluator. It reports
NLL/perplexity at every requested pass, per-source NLL at every pass, and RMS
changes in top-layer hidden states between successive passes.

Online recurrent generation for FBT, MemoryAdd, and MemoryTape32 is intentionally deferred
until their finite-pass training behavior has been validated. The current
`generate()` method on multipass wrappers delegates to vanilla generation.

## 11. External benchmark battery

The existing harness integration remains available for the one-pass control:

```bash
uv run python scripts/eval_lm.py \
  --config configs/mac/vanilla.yaml \
  --suite eval_configs/quick.yaml \
  --limit 100
```

The benchmark battery is secondary to held-out Dolmino NLL during the wiring
stage.

## Repository layout

```text
src/tiny_mistral/                   frozen vanilla backbone
src/tiny_mistral_mptt/attention/    research-only local memory attention
src/tiny_mistral_mptt/data/         Dolmino materialization/mmap dataset
src/tiny_mistral_mptt/training/     phases, pass/LR schedules, checkpointing
src/tiny_mistral_mptt/variants/     vanilla, FBT, MemoryAdd, MemoryTape32
src/tiny_mistral_mptt/evaluation/   one-pass NLL, pass-depth, lm-eval adapter
configs/                            data/Mac/GPU experiment configs
eval_configs/                       external benchmark suites
docs/                               protocol, data, provenance, validation
tests/                              offline unit/contract tests
```

See `docs/UPSTREAMS.md` for provenance and `docs/VALIDATION.md` for the tested
and still-pending hardware gates.
