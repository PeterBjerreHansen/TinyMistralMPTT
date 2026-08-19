# Efficiency benchmarks

These benchmarks characterize implementation efficiency: wall time, throughput,
memory use, precision mode, and feasible context/batch sizes. They are
**engineering measurements**, not scientific model-quality evidence, and do not
belong under `benchmarks/development/` or `benchmarks/core/`.

The shared runner performs real forward/backward/AdamW steps on deterministic
synthetic token IDs so data loading and storage do not contaminate the
measurement. The same training, context-scaling, and batch-scaling suites run on
MPS or CUDA via a device override. Precision suites remain backend-specific
because BF16 support is capability-dependent.

The default training, batch-scaling, and precision suites use the intended
2048-token context. The context-scaling suite also retains 512, 4096, and 8192
as explicit comparison points; those are not default training settings.

## MPS

```bash
uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/training.yaml \
  --device mps \
  --output benchmarks/efficiency/results/mps_training.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/precision_mps.yaml \
  --output benchmarks/efficiency/results/mps_precision.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/context_scaling.yaml \
  --device mps \
  --output benchmarks/efficiency/results/mps_context.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/batch_scaling.yaml \
  --device mps \
  --output benchmarks/efficiency/results/mps_batch.json
```

MPS context and batch scaling use FP32 compute by default. If the precision
suite shows that BF16 autocast is supported and numerically healthy on the
specific Mac/PyTorch stack, the general suites can also be rerun with
`--autocast-dtype bfloat16`.

## CUDA

```bash
uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/training.yaml \
  --device cuda \
  --output benchmarks/efficiency/results/cuda_training.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/precision_cuda.yaml \
  --output benchmarks/efficiency/results/cuda_precision.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/context_scaling.yaml \
  --device cuda --autocast-dtype bfloat16 \
  --output benchmarks/efficiency/results/cuda_context.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/batch_scaling.yaml \
  --device cuda --autocast-dtype bfloat16 \
  --output benchmarks/efficiency/results/cuda_batch.json
```

Each successful row records unique tokens/s, pass-tokens/s, milliseconds/step,
projected hours per 100M unique tokens, parameter/gradient/optimizer-state
dtypes, and available memory telemetry. CUDA reports peak allocated/reserved
memory. MPS currently reports allocator/driver memory at the end of the measured
window because PyTorch does not expose the same peak API.

A case that exceeds memory is recorded as `status: oom` and the suite continues.
An unavailable BF16 mode is recorded as `status: unsupported` rather than
invalidating the whole precision suite.

Compact retained results live under `benchmarks/efficiency/results/`. See
`docs/PRECISION.md` for the training precision contract.
