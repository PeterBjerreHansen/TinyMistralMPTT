# Precision contract

`dtype` is the **learned-parameter storage dtype**. Existing MPS development
runs remain FP32.

For serious CUDA training the intended mode is:

```yaml
device: cuda
dtype: float32
autocast_dtype: bfloat16
```

This keeps learned parameters and ordinary AdamW state in FP32 while executing
eligible CUDA operations under BF16 autocast. BF16 here is a training compute
format, not INT8/INT4 inference quantization.

MPS uses FP32 parameter storage and FP32 compute by default. The efficiency
battery also tests:

```yaml
device: mps
dtype: float32
autocast_dtype: bfloat16
```

MPS BF16 autocast is treated as a **capability-dependent engineering mode**:
PyTorch/macOS support can vary by host stack. The precision benchmark records it
as `unsupported` if the requested mode is unavailable. It should not be promoted
to a scientific training protocol until the local precision comparison is finite
and stable.

The experiment configuration deliberately allows only BF16 autocast for now.
FP16 training would require a separately validated loss-scaling policy and is
not silently enabled.

Pure BF16 parameter storage is not the intended training contract. The key
comparison is FP32 parameters/optimizer state with either FP32 compute or BF16
autocast compute.
