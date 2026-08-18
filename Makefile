.PHONY: test compile check download verify hf-check hf-layers hf-embeds \
  mps-smoke prepare-data verify-data evaluate-nll evaluate-quick substrate-gates cleanroom-gates

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
