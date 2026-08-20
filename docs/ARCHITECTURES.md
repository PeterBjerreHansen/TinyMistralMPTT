# Architecture contracts

Experiment-specific learning rates, token budgets, pass depth, initialization,
and checkpoint ancestry belong under `benchmarks/`. This file defines only the
reusable architecture surface. The full tape/control-token contract is in
`TAPE_MEMORY.md`.

## Vanilla

One ordinary TinyMistral causal pass with no architecture-added parameters.

## FBT

An independent multipass comparison based on asymmetric latent feedback. It is
not part of the tape family.

## MemoryAdd

For pass `k > 1`, current token embedding `e_t` receives the immediately
preceding previous-pass top state:

```text
x_t = e_t + W_A RMSNorm(h^(k-1)_(t-1))
```

`W_A` is bias-free and zero-initialized. Position zero receives a zero recurrent
residual.

## Tape

`TapeVariant` has one shared identity-initialized bias-free writer

```text
m = W_write h
```

and one independent GQA tape reader per decoder layer. Every reader consumes the
same previous-pass top-layer tape. Within a current-pass decoder layer the tape
residual is applied after the ordinary self-attention residual and before the
MLP.

The architecture has three write policies:

- `dense`: write every ordinary position;
- `periodic`: write positions satisfying `(t + 1) % C == 0`;
- `memory_token`: write only explicit input-only `<MEM>` positions.

`memory_window=W` counts committed tape records, not source-token distance.
Every tape read is strict-past: a record written at physical position `t` is
first available to position `t+1`.

Dense and periodic C=1 are the same implementation and are required to be
numerically identical with matching weights.

## TapeAddHybrid

`TapeAddHybridVariant` is the same tape plus the MemoryAdd path. There is no
gate, controller, or fusion MLP between the channels.

For ordinary-token sequences its fast path is ordinary MemoryAdd. With explicit
memory slots, `<MEM>` does not advance the fast state. For

```text
A <MEM> B
```

previous-stream `h_A` supplies the Add residual to both `<MEM>` and B;
`h_MEM` writes the slow tape; B then becomes the next fast state. This preserves
a clean distinction between ordinary-token fast recurrence and explicit tape
writes.

## Shared multipass causal invariant

Pass 1 is the current TinyMistral stream. Pass `k>1` consumes a completed
previous-pass top-state sequence, allowing sequence-parallel training. Exact
cached K-pass inference snapshots lower-stream feedback before computing the
same physical position in higher streams, so no same-position lower-stream
state leaks upward. Collapsed recurrent inference closes the final stream only
after the exact K-pass prefill boundary.
