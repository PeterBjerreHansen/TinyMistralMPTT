# Local NMP probe results

All six arms completed on MPS from clean commit `7055de1`. Each arm trained for
131,072 linguistic tokens (64 optimizer steps), retained one final checkpoint,
and evaluated eight held-out blocks at four pass depths. The run journals and
checkpoints remain local under `results/generated/`.

## Matched outcomes

| Pair | Control validation NLL | NMP validation NLL | NMP minus control | Late mean NTP-loss delta |
| --- | ---: | ---: | ---: | ---: |
| MemoryAdd recurrent NMP | 2.593085 | 2.593044 | -0.000041 | -0.000024 |
| Periodic-C32 Tape NMP | 2.473855 | 2.473856 | +0.000002 | -0.000027 |
| Recirculation–Tape dual NMP | 2.451247 | 2.451333 | +0.000086 | -0.000026 |

These differences are negligible and far below what one short seed can support
as a quality claim. The important result is that NTP behavior remains matched
while the new heads learn their auxiliary targets.

## Auxiliary learning

| Arm/objective | Mean first 8 steps | Mean last 8 steps | Change | Late weighted contribution |
| --- | ---: | ---: | ---: | ---: |
| MemoryAdd recurrent | 0.5931 | 0.5255 | -11.4% | 0.0271 |
| Periodic Tape | 0.7009 | 0.5293 | -24.5% | 0.0533 |
| Hybrid recurrent | 11.7067 | 10.9715 | -6.3% | 0.1125 |
| Hybrid Tape | 0.8433 | 0.6034 | -28.4% | 0.0634 |

Tape prediction is the clearest short-run learning signal. Its recent error is
horizon ordered in both tape arms: prediction is best one token before a write
and becomes gradually worse through the 17–32-token bucket. For periodic Tape,
recent bucket losses range from 0.522 at distance 1 to 0.538 at distance 17–32.
For the hybrid, they range from 0.589 to 0.615.

Pass-specific NMP losses are nearly equal on the seven K=3 batches. At this
dose there is no evidence that deeper passes anticipate the final-pass target
better. This is a result to measure again in a longer continuation, not a
reason to change pass weighting now.

The objectives have different natural scales. Final target RMS was 1.24 for
MemoryAdd, 1.39 for periodic Tape, 1.58 for hybrid Tape, and 19.28 for the
hybrid's internal recirculation source. The scale-aware coefficients kept every
auxiliary term subordinate to NTP, but raw recurrent and tape losses must not be
compared directly.

## Stability and cost

Every loss and gradient was finite. Mean/max gradient norms were:

| Arm | Mean | Maximum |
| --- | ---: | ---: |
| MemoryAdd control | 7.341 | 21.537 |
| MemoryAdd NMP | 7.330 | 20.989 |
| Tape control | 1.172 | 1.782 |
| Tape NMP | 1.170 | 1.782 |
| Hybrid control | 1.157 | 1.624 |
| Hybrid dual NMP | 1.157 | 1.624 |

Prediction heads reduced mean throughput by about 3.1% for MemoryAdd, 3.5% for
Tape, and 6.1% for the dual-head hybrid. This is consistent with one versus two
training-only MLP heads.

## Interpretation and next experiment

The implementation passes its local empirical gate: it learns causally aligned
targets, preserves NTP, initializes cleanly from NTP-only checkpoints, and has
modest overhead. This probe does not show an NLL benefit.

If the objective is promoted, the next useful experiment is a longer paired
continuation for periodic Tape and its control. Tape has the strongest learning
curve and the cleanest horizon diagnostic. A 1M-token run using the same weight
`0.10` is more informative than adding a weight sweep. The dual objective should
wait until that result; if promoted later, retain recurrent weight `0.01` rather
than increasing it in response to the larger raw loss.
