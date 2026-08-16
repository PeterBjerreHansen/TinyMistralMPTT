# Validation record

This document separates the already validated vanilla substrate, the multipass
experiments already exercised on the target Mac, and the new MemoryAdd code that
still requires its first full Mac Phase-A run.

## Vanilla Mac acceptance (2026-08-16)

The repository bootstrap was exercised on the target Mac/MPS development
machine before the multipass models were added:

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

These results validate the data/trainer/evaluation substrate.

## Multipass Mac results already obtained

The `mptt_v1` branch has also been exercised on the target Mac for the first FBT
and MemoryTape32 experiments. These are experimental observations rather than
claims of tuned final performance.

### MemoryTape32

Frozen-backbone Phase A completed successfully. Full pass-depth validation was:

```text
pass 1: 2.6645
pass 2: 2.6056
pass 3: 2.6065
pass 4: 2.6063
```

A causal intervention confirmed that the actual previous-pass memory content is
useful:

```text
real previous-pass memory: 2.6056
zeroed memory:             2.6645
mismatched memory:         2.6430
```

This is the current strongest finite-pass retrofit result in the repository.

### FBT

The original frozen-backbone Phase A was unstable after the vanilla pass:

```text
2.6645 -> 6.3028 -> 8.4157 -> 8.1146
```

A 262,144-token prefix-free co-adaptation run with pretrained LR `1e-7`, added
LR `1e-5`, and fixed two-pass training improved pass-2 NLL monotonically but
remained far from vanilla. A subsequent calibrated-initialization run matched
fused-input RMS to token-embedding RMS and set gate-logit standard deviation to
1.0. It improved full pass-2 NLL from 5.886 to 5.575, but still did not justify
pass-3+ training. FBT remains a valid implemented comparison, but it is not the
current reference for recurrent-system development.

## Offline acceptance after adding MemoryAdd

The implementation host is Linux/CPU and has no MPS/CUDA runtime. The complete
offline suite after adding MemoryAdd passes:

```text
99 passed, 4 skipped
```

All four skips are MPS hardware-only: the inherited MPS smoke plus one MPS
forward/backward case each for FBT, MemoryAdd, and MemoryTape32.

Coverage includes all previous vanilla/FBT/MemoryTape32 tests plus the following
MemoryAdd-specific contracts:

- one-pass exact vanilla parity;
- zero-initialized passes 1-4 are all exactly vanilla;
- strict one-token previous-pass alignment with an exact zero residual at
  position zero;
- shared causal shift semantics with FBT;
- zero previous state remains an exact vanilla input path even after the reader
  projection changes;
- Phase A freezes the full pretrained backbone while the zero-initialized
  memory projection receives a finite nonzero gradient;
- Phase B allows gradients into the pretrained backbone;
- MemoryAdd construction does not advance the global PyTorch RNG;
- factory/config registration and dtype propagation;
- strict state-dict roundtrip of added parameters;
- full shared-trainer Phase-A smoke with correct unique/token-equivalent compute
  accounting and unchanged pretrained weights;
- MPS forward/backward coverage is present and will run automatically on Apple
  hardware.

Compilation also passes with:

```bash
python -m compileall -q src scripts tests
```

## Required Mac gate for MemoryAdd

Before starting MemoryAdd Phase B, run:

```bash
uv run pytest -q

uv run python scripts/train.py \
  --config configs/mac/memory_add_phase_a.yaml

uv run python scripts/eval_pass_depth.py \
  --config configs/mac/memory_add_phase_a.yaml \
  --checkpoint runs/mac-memory-add-phase-a/latest.pt \
  --passes 8

uv run python scripts/eval_memory_interventions.py \
  --config configs/mac/memory_add_phase_a.yaml \
  --checkpoint runs/mac-memory-add-phase-a/latest.pt
```

The acceptance questions are deliberately scientific rather than merely
operational:

1. Does pass 1 stay at the vanilla NLL?
2. Does pass 2 improve materially below vanilla under a frozen backbone?
3. Does real previous state beat both zero and mismatched state?
4. Do passes 3-8 remain finite and reasonably stable?
5. Is the learned memory residual nontrivial relative to the token embedding,
   rather than remaining effectively bypassed?

Only if these gates are healthy should the checked-in conservative Phase-B
starting point be run:

```bash
uv run python scripts/train.py \
  --config configs/mac/memory_add_phase_b.yaml
```

## CUDA gate

No CUDA efficiency claim is validated in this phase. Before a rented-GPU run,
benchmark vanilla, MemoryAdd, FBT, and MemoryTape32 at the intended context
length and record peak memory, tokens/sec, parameter count, and effective pass
count. Recurrent decoding is intentionally still deferred until the finite-pass
MemoryAdd gate has been inspected.
