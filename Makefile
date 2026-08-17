.PHONY: test compile download verify hf-check hf-layers hf-embeds mps-smoke prepare-dev verify-data train-mac eval-nll eval-quick offline-check mac-gates fbt-a fbt-b memory-a memory-b memoryadd-a memoryadd-b fbt-depth memory-depth memoryadd-depth memoryadd-interventions recurrent-eval

test:
	uv run pytest -q

compile:
	uv run python -m compileall -q src scripts tests

offline-check: compile test

download:
	uv run python scripts/download_checkpoint.py

verify:
	uv run python scripts/verify_checkpoint.py

hf-check:
	uv run python scripts/compare_hf.py --device cpu --dtype float32

hf-layers:
	uv run python scripts/compare_hf_layers.py --device cpu --dtype float32

hf-embeds:
	uv run python scripts/compare_hf_inputs_embeds.py --length 40

mps-smoke:
	uv run python scripts/mps_smoke.py

prepare-dev:
	uv run python scripts/prepare_data.py --config configs/data/dolmino_dev_512.yaml

verify-data:
	uv run python scripts/verify_data.py data/dolmino/dev_512

train-mac:
	uv run python scripts/train.py --config configs/mac/vanilla.yaml

fbt-a:
	uv run python scripts/train.py --config configs/mac/fbt_phase_a.yaml

fbt-b:
	uv run python scripts/train.py --config configs/mac/fbt_phase_b.yaml

memory-a:
	uv run python scripts/train.py --config configs/mac/memory_tape32_phase_a.yaml

memory-b:
	uv run python scripts/train.py --config configs/mac/memory_tape32_phase_b.yaml

memoryadd-a:
	uv run python scripts/train.py --config configs/mac/memory_add_phase_a.yaml

memoryadd-b:
	uv run python scripts/train.py --config configs/mac/memory_add_phase_b.yaml

memoryadd-depth:
	uv run python scripts/eval_pass_depth.py --config configs/mac/memory_add_phase_a.yaml --checkpoint runs/mac-memory-add-phase-a/latest.pt --passes 8

memoryadd-interventions:
	uv run python scripts/eval_memory_interventions.py --config configs/mac/memory_add_phase_a.yaml --checkpoint runs/mac-memory-add-phase-a/latest.pt

fbt-depth:
	uv run python scripts/eval_pass_depth.py --config configs/mac/fbt_phase_b.yaml --checkpoint runs/mac-fbt-phase-b/latest.pt --passes 8

memory-depth:
	uv run python scripts/eval_pass_depth.py --config configs/mac/memory_tape32_phase_b.yaml --checkpoint runs/mac-memory-tape32-phase-b/latest.pt --passes 8

eval-nll:
	uv run python scripts/eval_nll.py --config configs/mac/vanilla.yaml

eval-quick:
	uv run python scripts/eval_lm.py --config configs/mac/vanilla.yaml --suite eval_configs/quick.yaml --limit 100

# Assumes the checkpoint and development data artifact have already been prepared.
mac-gates: test verify hf-check hf-layers hf-embeds mps-smoke verify-data eval-nll

recurrent-eval:
	uv run python scripts/eval_recurrent.py --config configs/mac/memory_add_phase_b_selected_lr1e-7_long.yaml --checkpoint runs/mac-memory-add-phase-b-selected-lr1e-7-long/latest.pt --prefill-passes 2 --prompt-tokens 256 --continuation-tokens 256
