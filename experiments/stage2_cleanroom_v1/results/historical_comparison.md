# Clean-room versus historical runs

This note compares the locked `stage2_cleanroom_v1` results with the earlier
Stage 1/Stage 2 runs. Historical results remain useful, but they do not all
have the same causal status.

The raw historical run directories, checkpoints, and development configs were
removed after this audit. The tables below preserve the conclusions needed for
the protocol decision without keeping the obsolete experiment surface active.

## Bottom line

- The clean E1 MemoryAdd and MemoryTape32 checkpoints reproduce the historical
  frozen-wired model weights exactly, bit-for-bit. The checkpoint-file hashes
  differ because the package metadata, optimizer state, and RNG state differ.
- The clean-room data manifest is identical to the historical
  `data/dolmino/dev_512` manifest. Dataset drift is therefore not explaining
  the result differences.
- The old `1e-7` Stage 2 runs are valid historical evidence, but the clean
  protocol's `3e-7` runs are the current prospective results. Their improvement
  is not a pure learning-rate ablation because the old runs and clean runs have
  different training trajectories and source-code provenance.
- The old K=3 runs were descendants of already jointly adapted K=2 models. The
  clean K=3 runs start independently from E1 and include a compute-matched K=2
  control, so they are the first controlled K comparison.

## Closest learning-rate comparison

Full 256-block validation, pass 2 NLL:

| variant | historical `1e-7`, 262k | clean `3e-7`, 262k | difference |
|---|---:|---:|---:|
| MemoryAdd | 2.567687 | 2.561092 | -0.006595 |
| MemoryTape32 | 2.552113 | 2.547850 | -0.004263 |

This is the closest comparison because both arms use the frozen-wired starting
point, the same data manifest, and the same 262k-token dose. It is still not a
strict source-level replay: the old `run.json` files do not record a source
revision, and the clean run uses the current code and a fresh checkpoint
package.

At the one-million-token endpoint, the historical selected-`1e-7` runs reached
pass 2/pass 8 values of `2.554071 / 2.563780` for MemoryAdd and
`2.543211 / 2.547322` for MemoryTape32. The clean `3e-7` K=2 runs reached
`2.535136 / 2.543786` and `2.529429 / 2.533141`, respectively. These are
encouraging improvements, but should be described as protocol-level results,
not as an isolated LR effect.

## Pass-depth comparison

The old K=3 descendants were useful health runs:

| variant | old K=3 pass 2 | old pass 3 | old pass 8 |
|---|---:|---:|---:|
| MemoryAdd | 2.5518 | 2.5536 | 2.5557 |
| MemoryTape32 | 2.5408 | 2.5460 | 2.5436 |

The independent clean full-dose K=3 reruns with `[0.05,0.20,0.75]` reached:

| variant | clean K=3 pass 2 | clean pass 3 | clean pass 8 |
|---|---:|---:|---:|
| MemoryAdd | 2.537559 | 2.536231 | 2.539310 |
| MemoryTape32 | 2.530788 | 2.531983 | 2.531653 |

At equal one-million-token dose, clean K=3 improves pass 8 over clean K=2 by
`0.004476` for MemoryAdd and `0.001488` for MemoryTape32. At matched
pass-token compute, K=2 remains better than K=3 by `0.004704` and `0.006397`,
respectively. This is why both controls should remain in the interpretation.

## FBT comparison

FBT is informative as a rejected retrofit, not as a numerically matched main
arm. Its historical 262k runs report only 16 validation blocks and begin with
a strongly degraded pass-2 state:

| FBT run | pass 1 | pass 2 |
|---|---:|---:|
| phase A | 2.578057 | 6.209366 |
| calibrated initialization | 2.577200 | 5.420014 |
| prefix-free co-adaptation | 2.577700 | 5.709383 |
| prefix-mixed adaptation | 2.576927 | 4.623794 |

Calibration and adaptation helped, but none approached the roughly 2.55
pass-2 range of the validated MemoryAdd/Tape32 starting points. The run names
are still accurate; they should not be presented as failed versions of the
clean-room protocol because FBT has a different interface, parameter count,
initialization, and evaluation scope.

## Stale or potentially misleading names

These artifacts should be retained, but their status should remain explicit:

- `mac-memory-*-phase-b-selected-lr1e-7[-long]` means the historical selected
  LR trajectory, not the current locked protocol.
- `mac-memory-*-phase-b-k3-short` means a short K=3 development descendant
  initialized from the old jointly adapted K=2 run, not an independent K=3
  experiment.
- The `*_continue_1m.yaml` files for old K=3 runs and the provisional K=2
  configs describe proposed continuations; their referenced output directories
  do not exist, so they are not completed results.
- `fbt_phase_b.yaml` also describes an unexecuted planned output directory
  (`runs/mac-fbt-phase-b`). The executed FBT runs are phase A, adaptation,
  co-adaptation, and calibrated initialization.

The historical results are therefore not stale as evidence. The stale part is
only treating old protocol decisions—especially `1e-7` as the selected LR or
the K=3 descendants as controlled comparisons—as if they were still current.

## Current comparison rule

Use `experiments/stage2_cleanroom_v1/results/` for current protocol conclusions and
this file for the retained historical comparison. Future results must state the
starting checkpoint, source revision, data-manifest hash, unique-token dose,
pass depth, and validation-block count beside every comparison.
