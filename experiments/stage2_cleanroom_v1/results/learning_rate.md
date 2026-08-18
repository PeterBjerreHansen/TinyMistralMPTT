# E2 LR dose response

E2 used 262,144 fresh unique tokens per arm. Every arm started independently
from its corresponding clean E1 checkpoint. The added-parameter learning rate
was `1e-6`; only the backbone learning rate varied.

Full 256-block validation was run at the endpoint with two passes.

| variant | backbone LR | pass-1 NLL | pass-2 NLL |
| --- | ---: | ---: | ---: |
| MemoryAdd | 0 | 2.664531 | 2.571398 |
| MemoryAdd | 3e-8 | 2.662374 | 2.570254 |
| MemoryAdd | 1e-7 | 2.657510 | 2.567687 |
| MemoryAdd | 3e-7 | 2.644916 | 2.561092 |
| MemoryTape32 | 0 | 2.664531 | 2.554481 |
| MemoryTape32 | 3e-8 | 2.662637 | 2.553750 |
| MemoryTape32 | 1e-7 | 2.658384 | 2.552113 |
| MemoryTape32 | 3e-7 | 2.647413 | 2.547850 |

The common E3 backbone learning rate is therefore `3e-7`, which is the best
tested value for both variants. E3 remains independently initialized from E1;
no E2 checkpoint is used as an E3 parent.
