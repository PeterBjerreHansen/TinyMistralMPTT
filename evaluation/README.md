# Evaluation suites

Reusable `lm-evaluation-harness` task suites live here. They are not tied to a
single benchmark study, so the suite definitions are kept separate from both
training configs and data artifacts.

- `suites/quick.yaml`: small development sanity battery.
- `suites/full.yaml`: broader base-model battery aligned with the tasks used in
  the Full-Bandwidth Transformer comparison.

Evaluation output should be written beside the benchmark/checkpoint being
studied when it is worth retaining; otherwise use a temporary path.
