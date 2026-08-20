# Scripts

Scripts are thin command-line entry points. Data recipes live under `data/`,
evaluation suites under `evaluation/`, and experiment settings with their owning
benchmark study/control.

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

## Efficiency and cloud qualification

```text
benchmark_training_efficiency.py
select_cuda_batch.py
cloud_preflight.py
```

The efficiency runner performs real optimizer steps and reports linguistic-token
and physical-position throughput when they differ. Tape cases must state their
write policy explicitly. `select_cuda_batch.py` consumes the dedicated CUDA K=2
qualification and chooses the smallest common efficient MemoryAdd/dense-Tape
microbatch rather than assuming maximum feasible batch is scientifically valid.

`cloud_preflight.py` checks CUDA/model/data/source/run compatibility, persistent
storage, free space, and memory-token-expanded batching before a paid run.

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

Training/evaluation loaders automatically wrap ordinary packed artifacts with
`MemoryTokenPackedDataset` when `memory_write_mode: memory_token`. The stored
data remain ordinary linguistic IDs; the view inserts input-only control ID V at
load time.

Pass-depth, memory interventions, and recurrent-inference scripts are reusable
checkpoint diagnostics. `evaluate_memory_interventions.py` can independently
intervene on TapeAddHybrid's fast MemoryAdd source and slow tape source.

Public `generate.py`/model generation remain ordinary language generation. The
low-level recurrent API can consume explicit MEM control steps, but no sampler
silently schedules architecture control positions.
