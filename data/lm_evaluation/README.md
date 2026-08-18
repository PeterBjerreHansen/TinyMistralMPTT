# Language-model evaluation

Checked-in lm-evaluation-harness task suites for quick substrate checks and
the broader base-model battery live here. Results belong in `results/` when a
run is worth retaining.

Run the short suite with:

```bash
uv run python scripts/evaluate_lm_harness.py \
	--config benchmarks/controls/substrate/mac.yaml \
	--suite data/lm_evaluation/quick.yaml \
	--limit 100
```
