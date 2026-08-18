.PHONY: test compile check download verify hf-check hf-layers hf-embeds \
  mps-smoke prepare-data verify-data evaluate-nll evaluate-quick substrate-gates cleanroom-gates k-sweep-gates \
  efficiency-mps efficiency-cuda efficiency-mps-training efficiency-mps-precision \
  efficiency-mps-context efficiency-mps-batch efficiency-cuda-training \
  efficiency-cuda-precision efficiency-cuda-context efficiency-cuda-batch cloud-preflight

test:
	uv run pytest -q

compile:
	uv run python -m compileall -q src scripts tests experiments

check: test compile
	git diff --check

download:
	uv run python scripts/download_model.py

verify:
	uv run python scripts/verify_model.py

hf-check:
	uv run python scripts/compare_to_hf.py --device cpu --dtype float32

hf-layers:
	uv run python scripts/compare_to_hf_layers.py --device cpu --dtype float32

hf-embeds:
	uv run python scripts/compare_to_hf_inputs_embeds.py --length 40

mps-smoke:
	uv run python scripts/smoke_mps.py

prepare-data:
	uv run python scripts/prepare_data.py \
		--config experiments/stage2_cleanroom_v1/configs/data/artifact.yaml

verify-data:
	uv run python scripts/verify_data.py data/stage2_cleanroom_v1/sequence_512

evaluate-nll:
	uv run python scripts/evaluate_nll.py \
		--config configs/substrate/mac/vanilla.yaml

evaluate-quick:
	uv run python scripts/evaluate_lm_harness.py \
		--config configs/substrate/mac/vanilla.yaml \
		--suite configs/evaluation/quick.yaml --limit 100

substrate-gates: check verify hf-check hf-layers hf-embeds mps-smoke

# Assumes the pinned backbone and clean-room data artifact are present.
cleanroom-gates: check verify-data

# Requires a committed, clean source tree before starting the selected-LR K sweep.
k-sweep-gates: check
	uv run python scripts/verify_k_sweep.py

# Engineering-only efficiency characterization. Results are written under runs/.
efficiency-mps: efficiency-mps-training efficiency-mps-precision efficiency-mps-context efficiency-mps-batch

efficiency-cuda: efficiency-cuda-training efficiency-cuda-precision efficiency-cuda-context efficiency-cuda-batch

efficiency-mps-training:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite efficiency_benchmarks/suites/training.yaml --device mps \
		--output runs/efficiency/mps_training.json

efficiency-mps-precision:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite efficiency_benchmarks/suites/precision_mps.yaml \
		--output runs/efficiency/mps_precision.json

efficiency-mps-context:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite efficiency_benchmarks/suites/context_scaling.yaml --device mps \
		--output runs/efficiency/mps_context.json

efficiency-mps-batch:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite efficiency_benchmarks/suites/batch_scaling.yaml --device mps \
		--output runs/efficiency/mps_batch.json

efficiency-cuda-training:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite efficiency_benchmarks/suites/training.yaml --device cuda \
		--output runs/efficiency/cuda_training.json

efficiency-cuda-precision:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite efficiency_benchmarks/suites/precision_cuda.yaml \
		--output runs/efficiency/cuda_precision.json

efficiency-cuda-context:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite efficiency_benchmarks/suites/context_scaling.yaml --device cuda --autocast-dtype bfloat16 \
		--output runs/efficiency/cuda_context.json

efficiency-cuda-batch:
	uv run python scripts/benchmark_training_efficiency.py \
		--suite efficiency_benchmarks/suites/batch_scaling.yaml --device cuda --autocast-dtype bfloat16 \
		--output runs/efficiency/cuda_batch.json

# Usage: make cloud-preflight CONFIG=path/to/config.yaml
cloud-preflight:
	@test -n "$(CONFIG)" || (echo "CONFIG is required" && exit 2)
	uv run python scripts/cloud_preflight.py --config $(CONFIG)
