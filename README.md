# TinyMistralMPTT — vanilla experimental substrate

This bootstrap is a **self-contained continued-pretraining/evaluation repo for
vanilla TinyMistral**. It deliberately stops before FBT, sparse MemoryTape, or
hybrid model code is introduced.

The purpose is to finish the architecture-independent experiment machinery
first: exact vanilla provenance, deterministic Dolmino materialization, a
reproducible token-budget trainer with checkpoint/resume, held-out NLL, and an
`lm-evaluation-harness` adapter. The next development phase can then wire FBT
into a system that already works end to end.

## Current boundary

Implemented:

- vendored TinyMistral vanilla implementation and baseline tests;
- pinned `M4-ai/TinyMistral-248M-v3` checkpoint helpers;
- deterministic `allenai/dolmino-mix-1124` 50B-recipe materialization;
- fixed `uint16` unpadded training/validation blocks and manifests;
- token-budget continued pretraining on CPU/MPS/CUDA;
- exact block-sampler checkpoint/resume state;
- unique-token and token-equivalent compute accounting;
- held-out NLL/perplexity with source breakdown;
- optional `lm-eval==0.4.12` adapter and checked-in quick/full suites;
- Phase-A/Phase-B trainability hooks, with Phase A intentionally a no-op for
  vanilla.

Not implemented: FBT, MemoryTape, sparse memory, hybrid recurrence, pass
schedules, or architecture-specific Phase-A wiring.

## 1. Install

Install `uv`, then from the repository root:

```bash
uv sync --extra data --extra eval
uv run pytest -q
```

`.python-version` pins Python 3.13 for this project. `uv` will provision it when
needed. A lockfile is intentionally not fabricated in this archive because the
build host cannot resolve the uncached online extras. The first successful
internet-connected `uv sync` will resolve/create `uv.lock`; commit that file
before serious experiments.

## 2. Download and verify TinyMistral

```bash
uv run python scripts/download_checkpoint.py
uv run python scripts/verify_checkpoint.py
```

For the strongest vanilla oracle checks:

```bash
uv run python scripts/compare_hf.py --device cpu --dtype float32
uv run python scripts/compare_hf_layers.py --device cpu --dtype float32
uv run python scripts/compare_hf_inputs_embeds.py --length 40
uv run python scripts/mps_smoke.py
```

## 3. Materialize a small Dolmino development artifact

The checked-in development config requests 1,048,576 train tokens and
131,072 validation tokens at sequence length 512:

```bash
uv run python scripts/prepare_data.py \
  --config configs/data/dolmino_dev_512.yaml
uv run python scripts/verify_data.py data/dolmino/dev_512
```

The data itself is not checked into git. Preparation streams each Dolmino source
only long enough to satisfy its published-mixture quota, writes the fixed local
binary artifact, and records hashes/revisions in `manifest.json`.

See `docs/DATA.md` before changing the recipe.

## 4. Vanilla continued-pretraining on a Mac

```bash
uv run python scripts/train.py --config configs/mac/vanilla.yaml
```

The MPS config intentionally uses FP32 training, matching the numerical
acceptance result from the vanilla reference repo. The local attention backend
is selected automatically on MPS for unpadded fixed-length batches.

Resume exactly from the most recent checkpoint:

```bash
uv run python scripts/train.py \
  --config configs/mac/vanilla.yaml \
  --resume-from runs/mac-vanilla/latest.pt
```

## 5. Evaluate held-out language-model loss

Base checkpoint:

```bash
uv run python scripts/eval_nll.py \
  --config configs/mac/vanilla.yaml \
  --max-blocks 32
```

Continued-pretraining checkpoint:

```bash
uv run python scripts/eval_nll.py \
  --config configs/mac/vanilla.yaml \
  --checkpoint runs/mac-vanilla/latest.pt
```

This is the primary experiment metric.

## 6. Run the external benchmark battery

Development run:

```bash
uv run python scripts/eval_lm.py \
  --config configs/mac/vanilla.yaml \
  --suite eval_configs/quick.yaml \
  --limit 100
```

Full predefined battery:

```bash
uv run python scripts/eval_lm.py \
  --config configs/mac/vanilla.yaml \
  --suite eval_configs/full.yaml
```

The adapter is deliberately single-process and batch-size-one for now. That is
simple and auditable; GPU batching can be optimized later if benchmark runtime
becomes material.

## 7. GPU-scale prepared config

`configs/data/dolmino_gpu_2048.yaml` describes an approximately 100M-token, 2M-validation-token,
2048-context artifact. `configs/gpu/vanilla.yaml` is a starting CUDA config,
not a claimed tuned optimum. Benchmark memory/throughput on the rented GPU
before committing to the final context length or batch size.

## Repository layout

```text
src/tiny_mistral/                 vendored vanilla backbone
src/tiny_mistral_mptt/data/       Dolmino recipe/materialization/mmap dataset
src/tiny_mistral_mptt/training/   trainer, phases, schedules, checkpoint state
src/tiny_mistral_mptt/variants/   currently vanilla only
src/tiny_mistral_mptt/evaluation/ held-out NLL and lm-eval adapter
configs/                          data/Mac/GPU experiment configs
eval_configs/                     external benchmark suites
docs/                             protocol, data contract, provenance
tests/                            offline unit/contract tests
```

## Research provenance

See `docs/UPSTREAMS.md`. The vanilla source derives from the validated
`TinyMistralFork` implementation; no code from the older MPTT repositories is
merged at this stage. The provenance document also records one deliberate
non-numerical attention-dispatch hardening in the research copy rather than
claiming a bit-for-bit source fork. Build-host test coverage and the remaining
Mac/network integration gates are recorded in `docs/VALIDATION.md`.
