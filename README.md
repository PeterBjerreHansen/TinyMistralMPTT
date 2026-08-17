# TinyMistralMPTT

Research code for retrofitting latent-state recurrence and memory into the
validated `M4-ai/TinyMistral-248M-v3` backbone.

The project is organized around one eventual controlled comparison: start from
a common pretrained TinyMistral backbone, construct viable architecture-specific
memory interfaces, then train the selected models under one locked protocol.

## Models

- `vanilla`: exact one-pass TinyMistral control.
- `memory_add`: one previous top-layer state, injected as a learned additive
  residual while preserving the token embedding path.
- `memory_tape32`: per-layer GQA cross-attention to up to 32 strictly earlier
  top-layer states from the previous pass.
- `fbt`: exact asymmetric-GLU feedback comparison. It remains implemented and
  documented, but the exact retrofit has not produced a satisfactory canonical
  starting point for the main comparison.

For every multipass variant, pass 1 is the unchanged vanilla model. Research
modules are bypassed entirely on that pass.

## Research workflow

The repository distinguishes **research stages** from the trainer's
**Phase A / Phase B** mechanics:

```text
SUBSTRATE VALIDATION
    TinyMistral parity, data, training, cache behavior
        ↓
STAGE 1 — STARTING POINTS
    develop and wire memory interfaces with a common frozen backbone
        ↓
    select immutable architecture-specific starting checkpoints
        ↓
STAGE 2 — TRAINING AND COMPARISON
    protocol development: LR, K, loss weighting, recurrence diagnostics
        ↓
    LOCK PROTOCOL
        ↓
    main large-scale Vanilla / MemoryAdd / MemoryTape comparison
        ↓
CAPABILITY ABLATIONS
    recurrent-memory tasks, sparse long-range memory, eventual hybrid
```

Stage 1 and Stage 2 can both contain development experiments. The distinction
is what is being selected:

- Stage 1 selects **starting models**.
- Stage 2 selects and then executes the **training protocol**.

A newer development checkpoint is never promoted automatically. Canonical
Stage 1 checkpoints are recorded in
`experiments/stage1_starting_points/STARTING_POINTS.yaml`.

## Repository rules

```text
src/          mechanisms and reusable implementation
scripts/      generic operations
configs/      configurations currently treated as canonical
experiments/  scientific questions, development configs, results, decisions
docs/         durable architecture/training/inference contracts
tests/        invariants that must not silently break
```

In particular, Stage 2 development configs live under `experiments/` until the
protocol is locked. `configs/stage2/` intentionally contains no main-run model
YAMLs yet.

## Install and validate

```bash
uv sync --extra data --extra eval
uv run pytest -q
uv run python -m compileall -q src scripts tests experiments
```

Download and verify the pinned TinyMistral checkpoint:

```bash
uv run python scripts/download_checkpoint.py
uv run python scripts/verify_checkpoint.py
uv run python scripts/compare_hf.py --device cpu --dtype float32
uv run python scripts/compare_hf_layers.py --device cpu --dtype float32
uv run python scripts/compare_hf_inputs_embeds.py --length 40
```

`docs/UPSTREAMS.md` and `docs/VANILLA_SOURCE.sha256` record the source and
checkpoint provenance.

## Development data

```bash
uv run python scripts/prepare_data.py --config configs/data/dolmino_dev_512.yaml
uv run python scripts/verify_data.py data/dolmino/dev_512
```

The current development artifact contains 1,048,576 training tokens and 131,072
validation tokens at sequence length 512. It is a development resource, not the
intended final Stage 2 large-run dataset.

## Stage 1: reproduce the selected wired starting points

Canonical wiring configs:

```bash
uv run python scripts/train.py --config configs/stage1/mac/memory_add_wiring.yaml
uv run python scripts/train.py --config configs/stage1/mac/memory_tape32_wiring.yaml
```

The selected immutable checkpoints are:

```text
checkpoints/memory_add_frozen_wired_v1.pt
checkpoints/memory_tape32_frozen_wired_v1.pt
```

with SHA-256 hashes recorded in `STARTING_POINTS.yaml`. These checkpoints share
exactly the same pretrained TinyMistral backbone; only the added memory pathways
were trained during wiring.

FBT retrofit development lives under
`experiments/stage1_starting_points/fbt_retrofit/`.

## Stage 2: protocol development

The Stage 2 protocol is **not locked yet**. Existing evidence is preserved by
question rather than by chronology:

- `protocol_development/learning_rate/`: frozen controls, LR dose response,
  selected K=2 co-adaptation runs;
- `protocol_development/pass_depth/`: K=3 development runs and continuation
  configs;
- `protocol_development/recurrence/`: exact-K versus collapsed-recurrent health
  checks.

See `experiments/stage2_training/main/LOCKED_PROTOCOL.md` for the current lock
status and the conditions for promoting configs into `configs/stage2/`.

## Evaluation and diagnostics

Finite-pass depth:

```bash
uv run python scripts/eval_pass_depth.py \
  --config <experiment-config.yaml> \
  --checkpoint <checkpoint.pt> \
  --passes 8
```

Causal memory interventions:

```bash
uv run python scripts/eval_memory_interventions.py \
  --config <experiment-config.yaml> \
  --checkpoint <checkpoint.pt>
```

Teacher-forced exact/recurrent comparison:

```bash
uv run python scripts/eval_recurrent.py \
  --config <experiment-config.yaml> \
  --checkpoint <checkpoint.pt> \
  --prefill-passes 1 2 3 4 \
  --prompt-tokens 256 \
  --continuation-tokens 256
```

MemoryAdd and MemoryTape32 implement K-general KV-cached exact incremental and
collapsed recurrent inference. Public multipass `generate()` intentionally
retains vanilla semantics; recurrent behavior is opt-in through the explicit
inference API. See `docs/RECURRENT_INFERENCE.md`.

## Layout

```text
configs/substrate/                   validated vanilla development configs
configs/stage1/                      canonical wiring configs
configs/stage2/                      future locked main-run configs
experiments/substrate_validation/    substrate research record
experiments/stage1_starting_points/  wiring evidence + checkpoint manifest
experiments/stage2_training/         protocol development + eventual main run
experiments/capability_ablations/     downstream capability questions
src/tiny_mistral/                    validated vendored TinyMistral backbone
src/tiny_mistral_mptt/               memory, training, evaluation, inference
docs/                                durable contracts and provenance
tests/                               numerical, causal, cache, trajectory gates
```

Start with `experiments/README.md` for the experiment-record convention and
`docs/TRAINING.md` for the durable trainer semantics.
