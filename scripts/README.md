# Scripts

Scripts are thin command-line entry points. Data recipes belong beside their
artifacts under `data/`; reusable evaluation suites live under `evaluation/`;
experiment settings live with their owner under `benchmarks/`. Raw training
outputs belong under the owning study/control's `results/generated/` directory.

## Setup and validation

```text
download_model.py
verify_model.py
compare_to_hf.py
compare_to_hf_layers.py
compare_to_hf_inputs_embeds.py
prepare_data.py
verify_data.py
verify_study.py
smoke_mps.py
```

## Efficiency and CUDA qualification

```text
benchmark_training_efficiency.py
select_cuda_batch.py
cloud_preflight.py
```

`benchmark_training_efficiency.py` measures real optimizer steps and supports
explicit gradient accumulation. `select_cuda_batch.py` consumes the dedicated
CUDA K=2 qualification result and selects the smallest common efficient
MemoryAdd/MemoryTape32 microbatch without pretending that a larger optimizer
batch has already been scientifically validated.

## Training and evaluation

```text
train.py
evaluate_nll.py
evaluate_lm_harness.py
evaluate_pass_depth.py
evaluate_memory_interventions.py
evaluate_recurrent_inference.py
generate.py
```

Pass-depth, memory-intervention, and recurrent-inference programs are reusable
checkpoint diagnostics. Their existence does not imply a dedicated benchmark
study. Script names describe operations directly and never encode a historical
campaign.

Sparse-memory variants use the same training/pass-depth/recurrent scripts as the
dense models. `evaluate_memory_interventions.py` also decomposes the hybrid's
fast MemoryAdd source and sparse-tape source independently, and reports sparse
writer identity deviation, write-state RMS/delta, and input/output cosine
diagnostics. Token-triggered write mode is model-supported but requires an
explicit data/token protocol before it should be used for a real study.
