# Validation record

This document separates the validated vanilla substrate, the finite-pass memory
experiments already exercised on the target Mac, and the K-general cached
recurrent-inference code whose CPU/reference oracle tests are complete and whose
new MPS inference tests remain to be exercised on Apple hardware.

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

This was the first successful MemoryTape32 wiring result; the later mature
frozen and Phase-B comparisons are recorded below.

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

## Mature MemoryAdd / MemoryTape32 Mac results

Both memory architectures completed frozen-backbone wiring to 1,048,576 tokens,
followed by a controlled Phase-B backbone-LR dose response and matched
one-million-token joint/frozen continuations. The full results and checkpoint provenance are in
`MEMORY_WIRED_BENCHMARK.md` and `MEMORY_PHASE_B_DOSE_RESPONSE.md`.

At the final joint checkpoints (pretrained LR `1e-7`, added LR `1e-6`, K=2),
full 256-block validation was:

```text
MemoryAdd:     pass 1 2.6394, pass 2 2.5541, pass 8 2.5638
MemoryTape32:  pass 1 2.6426, pass 2 2.5432, pass 8 2.5473
```

Matched frozen controls ended at pass-2 NLL `2.5675` and `2.5515`, respectively.
Final real/zero/mismatched interventions remained causal-memory-sensitive:

```text
MemoryAdd joint:     real 2.5541, zero 2.6394, mismatch 2.6705
MemoryTape32 joint:  real 2.5432, zero 2.6426, mismatch 2.6190
```

The backbone relative L2 movement from the mature wired checkpoints was about
`1.47e-4` for MemoryAdd and `1.41e-4` for MemoryTape32. No K=3/multi-depth
training was introduced.

## K-general cached/recurrent inference acceptance

The new inference implementation was validated on the Linux/CPU implementation
host with:

```text
126 passed, 8 skipped
```

The eight skips are MPS-only tests because Apple hardware is unavailable on the
implementation host. Compilation also passes with:

```bash
python -m compileall -q src scripts tests
```

New oracle/contract coverage includes:

- exact cached incremental inference versus full finite-pass recomputation for
  MemoryAdd and MemoryTape32 at K in `{1,2,3,4}`;
- continuation beyond the TinyMistral self-attention sliding window, exercising
  absolute cache positions and W-1 retained self-attention keys;
- K=1 exact/recurrent equivalence to ordinary vanilla cached inference;
- K>1 recurrent seeding from pass K-1 and exact equality through the first
  processed continuation token;
- snapshot-before-update K-stream semantics, preventing same-position feedback
  leakage;
- MemoryAdd's one-vector recurrent state;
- MemoryTape32's bounded, ordered strict-past ring and cached memory-bank GQA
  reader;
- `memory_bank_attention` against a dense GQA reference and against the final
  query of the existing full-sequence strict-past local reader;
- teacher-forced recurrent evaluator K=1 identity and K=2 delayed-onset drift
  semantics;
- MPS smoke cases for MemoryAdd and MemoryTape32 at K=2 and K=3, including
  exact and recurrent cached decode.

The public multipass `generate()` method remains intentionally vanilla. The new
recurrent path is explicit through `tiny_mistral_mptt.inference` and
`scripts/eval_recurrent.py`.

The applied overlay was then validated in this target Mac checkout with:

```text
134 passed
26/26 Mac configs parsed
compileall: PASS
git diff --check: PASS
```

The first teacher-forced K=2 smoke evaluations also completed for both final
joint checkpoints using 2 validation blocks, a 16-token prompt, and a
16-token continuation. They reported exact, recurrent, and vanilla NLLs plus
hidden-state RMS/cosine drift. The full 256-block/256-token command remains a
separate target-hardware gate because it was impractically slow on this MPS
setup; it was stopped without changing any repository or checkpoint state.

### Required Mac inference gate

Before interpreting target-hardware recurrent results, run:

```bash
uv run pytest -q

uv run python scripts/eval_recurrent.py \
  --config configs/mac/memory_add_phase_b_selected_lr1e-7_long.yaml \
  --checkpoint runs/mac-memory-add-phase-b-selected-lr1e-7-long/latest.pt \
  --prefill-passes 2 \
  --prompt-tokens 256 \
  --continuation-tokens 256

uv run python scripts/eval_recurrent.py \
  --config configs/mac/memory_tape32_phase_b_selected_lr1e-7_long.yaml \
  --checkpoint runs/mac-memory-tape32-phase-b-selected-lr1e-7-long/latest.pt \
  --prefill-passes 2 \
  --prompt-tokens 256 \
  --continuation-tokens 256
```

K=2 is the first scientific evaluation because those checkpoints were trained
with fixed K=2. The evaluator may then sweep other positive prefill depths
without changing code.

## CUDA gate

No CUDA efficiency claim is validated in this phase. Before a rented-GPU run,
benchmark vanilla, MemoryAdd, FBT, and MemoryTape32 at the intended context
length and record peak memory, tokens/sec, parameter count, and effective pass
count. Free-running recurrent generation remains deferred until the explicit
teacher-forced exact-vs-recurrent gate has been inspected on target hardware.
