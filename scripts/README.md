# Scripts

Scripts are thin command-line entry points. Data recipes belong beside their
artifacts under `data/`; experiment settings belong in `benchmarks/`; generated
outputs belong in the relevant experiment's `results/` directory.

## Setup and validation

```text
download_model.py
verify_model.py
compare_to_hf.py
compare_to_hf_layers.py
compare_to_hf_inputs_embeds.py
prepare_data.py
verify_data.py
smoke_mps.py
```

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

The names describe the operation directly; no script name encodes a specific
historical experiment.
