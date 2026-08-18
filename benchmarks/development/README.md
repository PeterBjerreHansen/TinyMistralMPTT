# Development benchmarks

Use this directory for structured studies that inform the protocol without
being the final core campaign. Examples include learning-rate sweeps,
pass-stability checks, recurrent-inference diagnostics, and K-schedule
selection. Shared data recipes live beside their generated artifacts under
`data/`, not in this benchmark taxonomy.

Every study should keep its configs, protocol notes, and `results/` directory
together.

Current development studies:

- `wiring/`
- `learning_rate/`
- `k_selection/`
- `pass_stability/`
- `exact_vs_recurrent_inference/`
