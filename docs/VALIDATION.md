# Validation record

This document separates the already validated vanilla substrate from the new
multipass code that still requires Apple/CUDA integration runs.

## Vanilla Mac acceptance (2026-08-16)

The repository bootstrap was exercised on the target Mac/MPS development
machine before FBT/MemoryTape32 model code was added:

- `uv sync --extra data --extra eval` resolved successfully.
- Full local suite: 63 passed.
- Pinned TinyMistral checkpoint verification passed with 248,024,064 parameters
  and weights SHA-256
  `9432ee6e0681473a9ed513e43362d9911832f9a5c7faded76f46ec66c55a9d3b`.
- MPS local-window smoke: forward max absolute difference 0, mean difference 0,
  finite backward loss 5.536343.
- Pinned Dolmino development artifact materialized and checksum verification
  passed: 1,048,576 train tokens and 131,072 validation tokens at length 512.
- Vanilla continued pretraining completed 128 optimizer steps / 262,144 unique
  tokens.
- Full 256-block held-out NLL improved from 2.66453146 (base checkpoint) to
  2.61463786 after the short continued-pretraining run.
- The quick lm-evaluation-harness integration completed all six 20-example
  development tasks without model/evaluator failure.

These results validate the data/trainer/evaluation substrate, not the new
multipass architectures.

## Multipass offline acceptance in the implementation environment

The implementation host is Linux/CPU and has no model/dataset network access or
MPS/CUDA hardware. The complete offline suite after adding FBT, MemoryTape32,
flexible pass schedules/objectives, optimizer groups, and pass-depth evaluation
currently passes: 86 passed, 3 skipped (all skips are MPS hardware-only).

Coverage includes:

- all inherited vanilla model/cache/mask/generation/training tests;
- FBT one-pass exact vanilla parity;
- strict previous-position FBT shift semantics;
- FBT Phase-A gradients restricted to added parameters;
- strict-past O(T*W) memory attention against a dense GQA oracle;
- exclusion of current/future and too-old memory positions;
- MemoryTape32 one-pass exact vanilla parity;
- MemoryTape32 manual decoder equivalence when its memory residual is zero;
- MemoryTape32 Phase-A gradients restricted to memory-reader parameters;
- right-aligned normalized pass-loss weighting;
- staged/stateful pass-scheduler exact restore;
- constant/cosine/piecewise-linear LR machinery;
- independent pretrained/added optimizer LR groups;
- Phase-A token-equivalent compute accounting;
- `init_from` model-only transfer into a fresh Phase-B run;
- bit-exact interrupted/resumed FBT training under a stochastic 1/2/3-pass
  schedule;
- pass-depth NLL and hidden-state-delta evaluation contracts.

The three skips are the inherited Apple MPS smoke plus two new FBT/MemoryTape32 MPS forward/backward tests.

## Required Mac gates for the new variants

Before treating FBT or MemoryTape32 as experimentally ready, run on the Mac:

```bash
uv run pytest -q

uv run python scripts/train.py --config configs/mac/fbt_phase_a.yaml
uv run python scripts/eval_pass_depth.py \
  --config configs/mac/fbt_phase_a.yaml \
  --checkpoint runs/mac-fbt-phase-a/latest.pt \
  --passes 4

uv run python scripts/train.py --config configs/mac/memory_tape32_phase_a.yaml
uv run python scripts/eval_pass_depth.py \
  --config configs/mac/memory_tape32_phase_a.yaml \
  --checkpoint runs/mac-memory-tape32-phase-a/latest.pt \
  --passes 4
```

Check finite losses/gradients, MPS memory use, throughput, pass-1 NLL parity,
pass-2 behavior, and repeated-pass stability before starting Phase B.

If Phase A is healthy, then run the checked-in Phase-B starting configurations:

```bash
uv run python scripts/train.py --config configs/mac/fbt_phase_b.yaml
uv run python scripts/train.py --config configs/mac/memory_tape32_phase_b.yaml
```

The provided LR ratios, pass weights, and fixed-two-pass schedule are starting
points and should be treated as experimental configuration rather than frozen
protocol.

## CUDA gate

No CUDA efficiency claim is validated in this phase. Before a rented-GPU run,
benchmark the vanilla, FBT, and MemoryTape32 variants at the intended context
length and record peak memory, tokens/sec, and effective pass count. The local
MemoryTape reader uses ordinary PyTorch O(T*W) tensor operations on both MPS and
CUDA; it is intentionally not yet a bespoke CUDA kernel.
