# Architecture contracts

This document describes model mechanisms only. Experiment-specific LR, token
budgets, pass depth choices, and checkpoint ancestry belong under
`benchmarks/`.

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

## SparseMemoryTape

`SparseMemoryTape` preserves the **same per-layer MemoryTape reader** as
MemoryTape32. The reader remains a bias-free GQA cross-attention residual after
self-attention and before the MLP. The architecture changes only which previous-
pass top states become addressable tape records.

A minimal bias-free hidden-size writer is applied at write positions:

```text
previous-pass top hidden at a write position
                |
          identity-initialized
             Linear(D,D)
                |
          committed record
```

The writer is identity-initialized. Consequently `memory_write_stride: 1` with
matching reader weights is exactly the dense MemoryTape32 mechanism at
initialization. This C=1 bridge is a regression invariant.

Two write policies are implemented:

```yaml
memory_write_mode: periodic
memory_write_stride: 8
memory_window: 32
```

writes after positions 7, 15, 23, ... (zero-based), while:

```yaml
memory_write_mode: token
memory_token_id: <vocabulary id>
memory_window: 32
```

writes exactly when that input token is encountered. Token-triggered mode is an
architecture capability only; the present Dolmino recipe does not yet insert a
new `<MEM>` token or define a loss/data protocol for one.

`memory_window` counts **committed memory records**, not source tokens. For a
periodic cadence C and window W, the **nominal tape span** is `C * W` source
positions: C controls the write rate (`1/C`) and W controls the number of
addressable candidate records. This is a write-event coverage measure, not a
claim that the tape contains only information from those positions; each deep
hidden state may summarize a much longer causal prefix.

The full-sequence implementation compacts committed records and gathers only the
last W strictly-prior records for each query, retaining O(T*W) reader work. In
cached inference a fixed-capacity W-entry bank carries an explicit validity mask
so different batch examples may commit at different token positions.

The timing rule is read-compute-write: a record committed at position `t` is
invisible at `t` and first becomes readable at `t+1`.

## MemoryAddSparseTape

`MemoryAddSparseTape` is deliberately the direct sum of the existing MemoryAdd
connection and SparseMemoryTape. There is no controller, gate, fusion MLP, or
learned weighting between channels.

For pass `k > 1`, the token input receives the existing immediate recurrence:

```text
x_t = e_t + W_M RMSNorm(h^(k-1)_(t-1))
```

and every decoder layer independently reads the sparse tape through the same
MemoryTape reader described above. The fast state updates every token; the tape
updates only on a write event. Cached feedback therefore stores both the last
source-stream hidden state and the sparse tape bank.

For diagnostic decomposition, zeroing the fast projection reduces the hybrid
to SparseMemoryTape; zeroing the tape reader outputs reduces it to MemoryAdd;
zeroing both yields the vanilla feedback computation.
