# TinyMistralMPTT

Research code for multipass/recurrent TinyMistral experiments on a validated
TinyMistral backbone. The active recurrent model surface is deliberately small:

- `memory_add`: one-step previous-pass top-state feedback;
- `recirculation`: fixed-alpha or adaptive source-layer feedback into an earlier layer;
- `tape`: one learned tape architecture with dense, periodic, or explicit
  `<MEM>` writes;
- `tape_add_hybrid`: the same tape plus the MemoryAdd fast recurrent channel.

`fbt` remains an independent research control and `vanilla` is the ordinary
TinyMistral baseline.

## Repository map

- `src/tiny_mistral/`: validated vendored vanilla TinyMistral implementation.
- `src/tiny_mistral_mptt/`: research architectures, training, evaluation, and inference.
- `benchmarks/`: controls, development/core studies, historical evidence, and
  engineering efficiency measurements.
- `data/`: deterministic dataset recipes; generated artifacts are local/ignored.
- `evaluation/`: reusable evaluation-suite definitions.
- `docs/`: architecture, data, training, inference, cloud, and validation contracts.

There is intentionally no central `configs/` directory. Runnable settings live
with the study or asset that owns them. Development/core studies use
`STUDY.yaml` for the scientific question and comparison structure; runnable YAML
files remain the execution source of truth.

Raw checkpoints, `run.json`, `metrics.jsonl`, `segments.jsonl`, snapshots, and
other large execution artifacts belong under the owning study/control's `results/generated/` directory and are ignored by Git.

## Tape model

All tape policies share the same identity-initialized learned writer and GQA
readers at configurable decoder layers. Reader outputs are zero-initialized,
and sequence-anchored RoPE is the default:

```yaml
variant: tape
memory_window: 32
memory_write_mode: dense       # dense | periodic | memory_token
memory_layers: [3, 7]          # or: all
memory_position_encoding: rope # default; explicit ablation: none
```

Periodic writes additionally require `memory_write_stride`. Explicit memory
slots use:

```yaml
variant: tape
memory_window: 32
memory_write_mode: memory_token
memory_write_stride: 8
memory_token_visibility: visible   # visible | write_only
```

`<MEM>` is an input-only architecture position with ID equal to the base
vocabulary size `V`; it is not added to the LM output head. For physical input
`A <MEM> B`, the language target at A is B, the MEM position has no LM loss, and
`h_MEM` writes one tape record. See `docs/TAPE_MEMORY.md` for the exact attention,
loss, cached-inference, and hybrid contracts.

## Training and cloud execution

The trainer supports exact resume on interruptible/spot instances: durable
checkpoint generations, newest-corrupt fallback, metrics repair, source/data
provenance, wall-clock and token checkpoint triggers, SIGINT/SIGTERM graceful
checkpointing, and weights-only scientific snapshots. See `docs/CLOUD.md`.

Memory-token runs distinguish linguistic data dose from physical transformer
work:

```text
unique_tokens_seen       = linguistic/data tokens
model_positions_seen     = ordinary + <MEM> physical positions
token_equivalent_compute = model positions x effective passes
```

Learning-rate schedules and run token budgets use linguistic tokens. Throughput
telemetry reports both linguistic tokens/s and model positions/s.

## Current research status

No long-run core comparison is locked. The active compute-conscious program is
defined in `benchmarks/development/experimental_pipeline.md`: local frozen-
backbone wiring, local Phase-B smoke tests, a resumable cloud pilot, and
selected confirmation runs. It deliberately fixes periodic C32/W32 Tape with
reader layers `[3, 7]` for Hybrid, and compares dense, periodic-C32, and
explicit-`<MEM>`-C32 Tape without a spacing or placement sweep. FBT remains in
the implementation but is not part of this experiment program.

Historical benchmark results remain read-only evidence; they do not define the
active architecture API.

## Validate

```bash
uv sync --extra data --extra eval
make check
```

Without dependency installation, the source tree can also be tested in an
environment that already provides the locked dependencies with:

```bash
PYTHONPATH=src pytest -q
```

Prepare/verify the local data artifact with:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/verify_data.py data/dolmino/local_2048
```

Before paid CUDA training, qualify batching using `benchmarks/efficiency/`, then
run the provider-agnostic preflight described in `docs/CLOUD.md`.
