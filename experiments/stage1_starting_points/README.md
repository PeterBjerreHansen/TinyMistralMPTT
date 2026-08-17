# Stage 1 — starting points

## Question

Can each candidate memory interface be made useful while preserving a common
pretrained TinyMistral backbone, so Stage 2 begins from fair and auditable model
states?

## Selected starting points

`STARTING_POINTS.yaml` is the authoritative manifest. MemoryAdd and
MemoryTape32 use immutable mature Phase-A checkpoints whose TinyMistral backbone
remained frozen and therefore identical to the vanilla control.

`WIRED_BENCHMARK.md` records the full 256-block depth and intervention checks.
The two `*_wired_checkpoint.yaml` files preserve the exact continuation configs
that produced the immutable artifacts; `configs/stage1/` contains the canonical
wiring recipe.

## FBT

The exact FBT retrofit remains implemented and its development record is kept in
`fbt_retrofit/`, but it is not currently a selected main-comparison starting
point. MemoryAdd is the one-state recurrent stand-in for Stage 2.

## Decision

Stage 1 is sufficiently complete for MemoryAdd and MemoryTape32. Future Stage 2
protocol development should branch from the manifest starting points when a
controlled comparison is required, rather than chaining through whichever
development checkpoint is newest.
