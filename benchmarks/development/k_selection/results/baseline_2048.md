# 2048-token K-selection baseline

This result record is the active 2048-token pre-lock K-selection comparison.
It is kept next to the configs that define the study.

The completed short Phase-B endpoints were:

| arm | pass 1 | pass 2 | pass 3 | pass 8 |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 2.374178 | — | — | — |
| MemoryAdd K=2 | 2.394452 | 2.321490 | — | 2.329156 |
| MemoryAdd K=3 | 2.409729 | 2.323894 | 2.323861 | 2.325000 |
| MemoryTape32 K=2 | 2.400027 | 2.316835 | — | 2.320331 |
| MemoryTape32 K=3 | 2.412874 | 2.317872 | 2.319745 | 2.319152 |

Pass-depth, memory-intervention, and recurrent-inference diagnostics can be
regenerated from retained checkpoints with the reusable evaluation scripts. No
dedicated development study is required for those diagnostics, and no core
campaign is locked by this qualification alone.
