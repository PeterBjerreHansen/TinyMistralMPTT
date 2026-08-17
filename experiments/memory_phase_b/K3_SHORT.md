# Short K=3 continuation

This archived experiment continued the mature K=2 Phase-B checkpoints for
262,144 unique tokens on Apple MPS. It used fixed K=3 training, pass-loss
weights `[0.1, 0.3, 0.6]`, pretrained LR `1e-7`, added-parameter LR `1e-6`, and
a constant schedule.

The checkpoints are local ignored artifacts at:

- `runs/mac-memory-add-phase-b-k3-short/latest.pt`
- `runs/mac-memory-tape32-phase-b-k3-short/latest.pt`

Full 256-block validation after the continuation:

| variant | pass 1 | pass 2 | pass 3 | pass 8 |
| --- | ---: | ---: | ---: | ---: |
| MemoryAdd | 2.635115 | 2.551827 | 2.553646 | 2.555656 |
| MemoryTape32 | 2.638809 | 2.540751 | 2.546005 | 2.543649 |

Both remained bounded through pass 8 and improved modestly over their mature
K=2 checkpoints. Full real/zero/mismatched interventions were:

| variant | real | zero | mismatched |
| --- | ---: | ---: | ---: |
| MemoryAdd | 2.551827 | 2.635115 | 2.665853 |
| MemoryTape32 | 2.540751 | 2.638809 | 2.614541 |

Teacher-forced K=3 cached inference on an 8-block, 64-token continuation kept
the maximum exact/recurrent NLL gap below `0.0031`. The minimum hidden-state
cosine was `0.9813` for MemoryAdd and `0.9973` for MemoryTape32.
