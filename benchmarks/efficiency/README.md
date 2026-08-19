# Efficiency benchmarks

These benchmarks characterize implementation efficiency: wall time, throughput,
memory use, precision mode, feasible microbatch size, and context scaling. They
are **engineering measurements**, not scientific model-quality evidence, and do
not belong under `benchmarks/development/` or `benchmarks/core/`.

The shared runner performs real forward/backward/AdamW optimizer steps on
deterministic synthetic token IDs so data loading and storage do not contaminate
the measurement. It supports explicit gradient accumulation and reports both
hardware-facing microbatch quantities and optimizer-facing batch quantities.

For sequence length `T`:

```text
microbatch_tokens = batch_size * T
optimizer_batch_tokens = batch_size * grad_accum_steps * T
```

Changing `batch_size` can therefore be both an engineering change and a
scientific optimizer-batch change. The benchmark tooling reports the distinction
rather than hiding it.

## Serious CUDA batching qualification

The 2048-token development evidence was trained with `batch_size=1` and
`grad_accum_steps=1`, i.e. 2,048 unique tokens per optimizer update. The current
CUDA substrate config intentionally starts from that same optimizer-batch size.
It does **not** assume that 32k tokens/update is appropriate merely because a GPU
can process larger batches.

On the intended GPU, run:

```bash
make efficiency-cuda-batch-qualification
make select-cuda-batch \
  RESULT=benchmarks/efficiency/results/cuda_batch_qualification.json
```

The qualification suite tests K=2 MemoryAdd and MemoryTape32 at 2048 context,
FP32 parameter/optimizer storage, BF16 autocast, `grad_accum_steps=1`, and
microbatches 1/2/4/8. OOM cases are recorded rather than aborting the suite.

`select_cuda_batch.py` chooses the **smallest common successful microbatch** that
reaches at least 90% of each architecture's best throughput across all of that
architecture's feasible tested batches. The selected batch must still be common
to every requested architecture. This biases toward preserving the smaller
optimizer batch when extra batching buys little throughput. Its output explicitly
states whether the recommendation changes the 2,048-token reference optimizer
batch. If it does, that is a protocol question to qualify before locking a core
run.

Gradient accumulation should not be increased just to manufacture a larger
optimizer batch. Use it only when a scientifically chosen optimizer batch must
be implemented with a smaller hardware microbatch.

## General suites

The shared training, batch-scaling, and precision suites use 2048-token context.
The context-scaling suite also retains 512, 4096, and 8192 as explicit
engineering comparison points; those are not default training settings.

### MPS

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
  --output benchmarks/efficiency/results/generated/mps_context.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/batch_scaling.yaml \
  --device mps \
  --output benchmarks/efficiency/results/generated/mps_batch.json
```

MPS context and batch scaling use FP32 compute by default. If the precision
suite shows that BF16 autocast is supported and numerically healthy on the
specific Mac/PyTorch stack, the general suites can also be rerun with
`--autocast-dtype bfloat16`.

### CUDA

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
  --output benchmarks/efficiency/results/generated/cuda_context.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/batch_scaling.yaml \
  --device cuda --autocast-dtype bfloat16 \
  --output benchmarks/efficiency/results/generated/cuda_batch.json
```

Each successful row records unique tokens/s, pass-tokens/s, optimizer steps/s,
microbatches/s, milliseconds/optimizer-step, microbatch tokens,
optimizer-batch tokens, projected hours per 100M unique tokens,
parameter/gradient/optimizer-state dtypes, and available memory telemetry. CUDA
reports peak allocated/reserved memory. MPS currently reports allocator/driver
memory at the end of the measured window because PyTorch does not expose the
same peak API.

A case that exceeds memory is recorded as `status: oom` and the suite continues.
An unavailable BF16 mode is recorded as `status: unsupported` rather than
invalidating the whole precision suite.

Compact retained results live under `benchmarks/efficiency/results/`. See
`docs/PRECISION.md` for the training precision contract.
