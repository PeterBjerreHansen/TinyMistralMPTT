# Selected-LR K sweep

This is the clean-room K-schedule comparison after E2 selected `1e-6` for
both backbone and added parameters. All eight arms started from the matching
E1 wiring checkpoint and trained for 1,048,576 unique tokens with sequence
length 512, seed 1337, and the same optimizer settings.

K=2 used weights `[0.25, 0.75]`. K=3 used `[0.05, 0.20, 0.75]`. Mixed
schedules selected the corresponding vector by realized K.

## Completed arms

The endpoint validation below is the final 16-block validation emitted by the
trainer. Pass 3 is blank for fixed K=2.

| architecture | schedule | pass 1 NLL | pass 2 NLL | pass 3 NLL | realized K | compute |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| MemoryAdd | K=2 | 2.43975 | 2.38277 | — | 2048×K2 | 2,097,152 |
| MemoryAdd | 90% K2 / 10% K3 | 2.44026 | 2.38263 | 2.39089 | 1837×K2, 211×K3 | 2,205,184 |
| MemoryAdd | 50% K2 / 50% K3 | 2.44805 | 2.38420 | 2.38726 | 1034×K2, 1014×K3 | 2,616,320 |
| MemoryAdd | K=3 | 2.46895 | 2.38628 | 2.38405 | 2048×K3 | 3,145,728 |
| MemoryTape32 | K=2 | 2.44417 | 2.38735 | — | 2048×K2 | 2,097,152 |
| MemoryTape32 | 90% K2 / 10% K3 | 2.44596 | 2.38770 | 2.39599 | 1837×K2, 211×K3 | 2,205,184 |
| MemoryTape32 | 50% K2 / 50% K3 | 2.45379 | 2.38720 | 2.39126 | 1034×K2, 1014×K3 | 2,616,320 |
| MemoryTape32 | K=3 | 2.47226 | 2.38688 | 2.38891 | 2048×K3 | 3,145,728 |

The realized mixture histograms match the seeded schedule used by the earlier
clean-room controls. All runs reached the exact token budget without NaNs or
checkpoint errors.

## Full pass-depth diagnostic

These values use the same 256 validation blocks for every arm and evaluate
passes 1–8 after training. The final column is the hidden-state RMS change at
pass 8 relative to pass 1.

| architecture | schedule | pass 1 NLL | pass 2 NLL | pass 8 NLL | pass-8 hidden Δ RMS |
| --- | --- | ---: | ---: | ---: | ---: |
| MemoryAdd | K=2 | 2.54392 | 2.49615 | 2.50236 | 0.1524 |
| MemoryAdd | 90% K2 / 10% K3 | 2.54447 | 2.49617 | 2.50112 | 0.0210 |
| MemoryAdd | 50% K2 / 50% K3 | 2.55068 | 2.49700 | 2.49904 | 0.0090 |
| MemoryAdd | K=3 | 2.56664 | 2.49864 | 2.49992 | 0.0149 |
| MemoryTape32 | K=2 | 2.54934 | 2.49671 | 2.49969 | 0.0325 |
| MemoryTape32 | 90% K2 / 10% K3 | 2.55097 | 2.49690 | 2.49914 | 0.0126 |
| MemoryTape32 | 50% K2 / 50% K3 | 2.55826 | 2.49704 | 2.49815 | 0.0010 |
| MemoryTape32 | K=3 | 2.57373 | 2.49763 | 2.49810 | 0.00004 |

All arms are bounded through pass 8. Pass 2 is consistently the useful
improvement; later passes do not diverge, but neither architecture shows a
clear extra NLL benefit from training K=3 rather than K=2.

## Memory interventions

These checks use 256 validation blocks on the fixed-K2 and fixed-K3 arms. The
zero-memory condition reproduces pass 1 exactly. Real memory is better than
zero memory in all four cases and better than mismatched memory in all four.

| architecture | schedule | pass-1 / zero NLL | real NLL | mismatched NLL | real hidden Δ RMS |
| --- | --- | ---: | ---: | ---: | ---: |
| MemoryAdd | K=2 | 2.54392 | 2.49615 | 2.58220 | 0.7037 |
| MemoryAdd | K=3 | 2.56664 | 2.49864 | 2.59964 | 0.5930 |
| MemoryTape32 | K=2 | 2.54934 | 2.49671 | 2.55226 | 0.5618 |
| MemoryTape32 | K=3 | 2.57373 | 2.49763 | 2.56028 | 0.4635 |

MemoryAdd's learned residual RMS is 0.336× embedding RMS at K=2 and 0.344×
at K=3, so its recurrent channel is not bypassed. Tape32 K=3's mismatched
state still helps relative to zero, but it is substantially worse than the
real state; this is weaker causal discrimination than MemoryAdd's.

## Bounded recurrent inference

Teacher-forced exact-cached versus recurrent inference used eight validation
blocks, a 64-token prompt, and a 64-token continuation. At horizon 64, the
recurrent-vs-exact NLL gap and hidden cosine were:

| architecture | prefill K | exact NLL | recurrent NLL | gap | hidden cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| MemoryAdd | 2 | 1.90310 | 1.90401 | +0.00091 | 0.9311 |
| MemoryAdd | 3 | 1.90180 | 1.90380 | +0.00200 | 0.9788 |
| MemoryTape32 | 2 | 1.90163 | 1.90444 | +0.00281 | 0.9767 |
| MemoryTape32 | 3 | 1.89910 | 1.89933 | +0.00024 | 0.9997 |

The collapsed recurrent handoff remains close to exact finite-K inference over
this bounded continuation. Longer-horizon or generation-length evaluation is
still a separate question.

## Current interpretation

The selected-LR sweep is complete, but the protocol remains `k_schedule_pending`
until a schedule is explicitly locked. The evidence supports K=2 as the clean
baseline: it matches or slightly beats K=3 on pass-2 NLL, costs less compute,
and has stable pass-depth behavior. The mixed schedules do not show a clear
advantage over fixed K=2 at this budget. Tape32 has especially tight pass-8
stability, while MemoryAdd shows the clearer real-versus-mismatched memory
separation.

## Provenance note

Source metadata was added while the campaign was already running. The earliest
K=2 and mixture `run.json` files therefore predate the `source` field rather
than being backfilled with an inaccurate commit. MemoryAdd K=3 records the
pre-freeze dirty development state; Tape32 K=3 records the clean validated
commit. The change was manifest-only and did not alter model, optimizer,
scheduler, or loss mathematics. The current source tree passes the clean
K-sweep gate.

The numerical result record is retained; the runnable configs and generated
checkpoints were removed after the 2048-token qualification became active.
