# Architecture contracts

This document describes model mechanisms only. Experiment-specific LR, token
budgets, pass depth choices, and checkpoint ancestry belong under
`experiments/`.

## Vanilla

One ordinary TinyMistral causal pass. It has no architecture-added parameters.
For all multipass wrappers, pass 1 is exactly this model.

## FBT

For pass `k > 1`, let `e_t` be the token embedding and let
`h^(k-1)_(t-1)` be the immediately preceding top-layer state from the previous
pass. Position zero retains `e_0`; later positions use

```text
x_t = W_U h^(k-1)_(t-1) * sigmoid(W_G e_t)
```

with two bias-free hidden-size projections. The exact retrofit remains a useful
comparison but is not currently a selected Stage 1 starting point.

## MemoryAdd

Pass 1 is vanilla. For pass `k > 1`:

```text
x_t = e_t + W_M RMSNorm(h^(k-1)_(t-1))
```

Position zero receives an exact zero memory residual. `W_M` is bias-free and
zero-initialized, so every pass depth is exactly vanilla before training.
MemoryAdd deliberately reuses the previous top hidden state directly; it has no
separate learned writer. This makes it the clean one-state recurrent control.

## MemoryTape32

Pass 1 is vanilla. For pass `k > 1`, every decoder layer inserts a residual GQA
cross-attention reader after self-attention and before the MLP. At position `t`
the reader may access only

```text
max(0, t-W) ... t-1
```

from the previous pass's final top-layer states. `W=32` is the reference setting
and `memory_window` remains configurable. The local reader is O(T*W), not O(T²).

MemoryTape32 is an addressable short latent tape. It should not be described as
the eventual sparse long-range/landmark architecture; that remains a downstream
capability ablation.

## Shared causal invariant

No architecture may consume same-position or future state from a previous pass.
Single-state variants use an explicit one-position right shift. Cached exact
inference snapshots lower-pass memory before computing the new position.
