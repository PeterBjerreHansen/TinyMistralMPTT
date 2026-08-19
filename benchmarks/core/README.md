# Core benchmarks

This directory is reserved for larger, predeclared experiments that test the
project's central claims. It is intentionally empty until development evidence
justifies a locked campaign.

Before a core run, create a study directory containing:

```text
STUDY.yaml
<runnable arm configs>.yaml
results/
```

Set `status: locked` only after the scientific question, arms, comparison axes,
data artifact, initialization provenance, and execution configs have been
reviewed. `scripts/verify_study.py` checks that compared arms differ only on the
arm-local output path, declared `experimental_axes`, and explicit
`allowed_differences`.
