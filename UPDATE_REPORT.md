# MemoryAdd update report

Prepared against `PeterBjerreHansen/TinyMistralMPTT` branch `mptt_v1`, observed
head `43e618af1d386f95f6a545c35fd619c91e005ff5` on 2026-08-16.

## Added architecture

`memory_add` is a first-class `MultiPassVariant` with one previous-pass latent
per token. For pass `k>1`:

```text
r_0 = 0
r_t = h^(k-1)_(t-1)
x_t = e_t + W_M RMSNorm(r_t)
```

`W_M` is bias-free and zero-initialized. This guarantees that every pass depth
is exactly vanilla at initialization. The current TinyMistral control reuses the
previous final top-layer hidden state directly rather than adding the older
standalone prototype's separate memory-write head; this keeps recurrent
bandwidth aligned with FBT and isolates the fusion/reader mechanism.

## Integration

- registered `memory_add` in experiment config and model factory;
- added a shared strict one-token previous-state shift helper used by MemoryAdd
  and FBT;
- preserved FBT prefix-mixin and calibrated-initialization functionality;
- added Mac Phase-A and conservative Phase-B configs;
- generalized the memory intervention script to support MemoryAdd and
  MemoryTape32;
- added MemoryAdd scale diagnostics (residual RMS / embedding RMS);
- added Makefile targets and protocol/provenance/validation documentation.

## Validation performed in this artifact build

```text
pytest:     100 passed, 4 skipped
compileall: PASS
git diff --check: PASS
```

The four skips are MPS-hardware tests because the build environment is
Linux/CPU. The MPS test suite now contains a dedicated MemoryAdd two-pass
forward/backward case, so it will execute automatically on the target Mac.

MemoryAdd-specific coverage includes:

- exact pass-1 vanilla parity;
- exact vanilla fixed point through pass 4 at zero initialization;
- strict `t-1` previous-pass alignment and exact zero residual at position 0;
- exact vanilla behavior for a zero previous state even after changing the
  memory projection;
- Phase-A freezing and finite/nonzero projection gradient;
- Phase-B backbone gradient flow;
- no global RNG advancement during MemoryAdd construction;
- factory/config registration and dtype propagation;
- strict state-dict roundtrip;
- shared-trainer Phase-A smoke with unchanged pretrained weights and correct
  token-equivalent compute;
- pass-depth evaluator fixed-point behavior at zero initialization.

## Next target-Mac gate

Do not start MemoryAdd Phase B before inspecting the Phase-A results:

```bash
uv run pytest -q
uv run python scripts/train.py --config configs/mac/memory_add_phase_a.yaml
uv run python scripts/eval_pass_depth.py \
  --config configs/mac/memory_add_phase_a.yaml \
  --checkpoint runs/mac-memory-add-phase-a/latest.pt \
  --passes 8
uv run python scripts/eval_memory_interventions.py \
  --config configs/mac/memory_add_phase_a.yaml \
  --checkpoint runs/mac-memory-add-phase-a/latest.pt
```

The primary gates are: pass-1 parity, pass-2 improvement, real-state advantage
over zero/mismatched state, stability through pass 8, and a nontrivial learned
residual magnitude.

Large checkpoints, data artifacts, run directories, and `.git` metadata are not
included in the distributed source archive.
