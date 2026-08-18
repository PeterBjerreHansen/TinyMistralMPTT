# Historical K=2/K=3 mixture results

> Superseded protocol-development evidence. These runs used backbone LR
> `3e-7`; the later selected-LR rerun is summarized in `k_sweep.md`.

These four optional mixture arms were run on 2026-08-18 from the clean E1
wiring checkpoints. They used the then-current, now superseded rates:

- backbone learning rate: `3e-7`
- added-parameter learning rate: `1e-6`
- unique-token budget: `1,048,576`
- K-specific loss weights: K=2 `[0.25, 0.75]`; K=3 `[0.05, 0.20, 0.75]`
- endpoint validation: 16 blocks, three passes

The realized schedules and final metrics were:

| variant | mixture | realized K histogram | token-equivalent compute | pass-1 NLL | pass-2 NLL | pass-3 NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MemoryAdd | 50/50 | K2=1034, K3=1014 | 2,616,320 | 2.517994 | 2.429144 | 2.433491 |
| MemoryAdd | 90/10 | K2=1837, K3=211 | 2,205,184 | 2.512554 | 2.428589 | 2.439373 |
| MemoryTape32 | 50/50 | K2=1034, K3=1014 | 2,616,320 | 2.523469 | 2.427573 | 2.433063 |
| MemoryTape32 | 90/10 | K2=1837, K3=211 | 2,205,184 | 2.518238 | 2.427146 | 2.438399 |

All four runs reached the full unique-token budget, wrote `latest.pt`, and
preserved the pass schedule, realized histogram, K-specific loss weights,
data-manifest hash, optimizer state, sampler state, and RNG state in the
checkpoint. The 50/50 and 90/10 arms have identical realized histograms
across architectures because they use the same scheduler seed and schedule.

## Initial reading

At this endpoint, the 50/50 mixture is slightly better than 90/10 on pass 3
for both variants:

- MemoryAdd: `2.433491` vs `2.439373`.
- MemoryTape32: `2.433063` vs `2.438399`.

The 90/10 mixture uses about 16% less token-equivalent compute than 50/50 and
has essentially the same pass-2 NLL. This makes it a plausible efficiency
control, but not evidence that the mixture is better overall. The endpoint
validation uses only 16 blocks, so these numbers should not be compared as if
they were the full 256-block E3 pass-depth evaluation; use a common full
validation protocol before locking a preference.

The implementation and regression gate were also rechecked after the runs:

```text
147 passed
compileall: PASS
git diff --check: PASS
```
