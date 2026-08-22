# Ad-hoc benchmarks

Use this directory for one-off exploratory comparisons that do not define a
reusable protocol decision. The active Recirculation–Tape NMP program is kept
under `recirculation_tape_nmp/`: it contains the 10M NTP continuations, the
checkpoint-scale diagnostics, and the longer auxiliary-objective configs.

Generated checkpoints, journals, and diagnostic JSON files belong under the
owning experiment stage's `results/generated/` directory and are ignored by
Git. Do not create a shared `benchmarks/ad_hoc/results/` tree. Keep the compact
interpretation and exact command sequence in the relevant stage directory so a
promising ad-hoc result can later be promoted into a locked development study.
