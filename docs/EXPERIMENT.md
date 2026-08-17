# Experimental protocol: one-state feedback and MemoryTape32 phase

## Scope

This phase compares four first-class models on the same pretrained TinyMistral
backbone and deterministic Dolmino token stream:

1. `vanilla`
2. `fbt`
3. `memory_add`
4. `memory_tape32`

No chunked-memory or hybrid design is part of this phase. The purpose is to
establish clean finite-pass training behavior before adding further model
families.

## Architecture invariants

### Vanilla

One ordinary TinyMistral pass. It has no architecture-added parameters and no
Phase A.

### FBT

Pass one is vanilla. For pass `k>1`, top-layer states from pass `k-1` are shifted
one token to the right. At `t>0` the next input is the asymmetric GLU

```text
W_U h^(k-1)_(t-1) * sigmoid(W_G e_t)
```

and position zero retains `e_0`. The added modules are two bias-free
`hidden_size x hidden_size` projections.

### MemoryAdd

Pass one is vanilla. For pass `k>1`, the previous pass's final top-layer states
are shifted one token to the right and used as a learned additive input
residual:

```text
x_t = e_t + W_M RMSNorm(h^(k-1)_(t-1))
```

Position zero has no predecessor and therefore receives an exact zero residual.
The bias-free `hidden_size x hidden_size` projection `W_M` is zero-initialized,
so every pass depth is exactly vanilla at initialization. The current TinyMistral
variant deliberately reuses the previous top hidden state directly; unlike the
older standalone prototype it does not add a separate learned memory-write head.
This isolates one-state recurrent bandwidth from the FBT fusion operator.

### MemoryTape32

Pass one is vanilla. For pass `k>1`, each decoder layer has a separate residual
GQA cross-attention reader over the previous pass's final top-layer states. A
query at token `t` may read only

```text
max(0, t-32) ... t-1
```

with no same-position previous-pass state. The default reader geometry mirrors
TinyMistral: 32 query heads, 8 KV heads, head dimension 32, no projection
biases. The reader is implemented with O(T*W) local score work and does not
construct a T x T memory score matrix.

The model name denotes the reference/default `W=32`; `memory_window` remains a
configurable experimental parameter.

## Pass 1 parity gate

For all multipass architectures:

```text
variant.compute_passes(ids, passes=1).pass[0]
```

must be numerically identical to direct vanilla TinyMistral. Research modules
are skipped, not merely multiplied by a zero gate.

`src/tiny_mistral/` is not modified to implement any research architecture.

## Flexible pass objective

The trainer does not hard-code the FBT paper's objective. Given pass losses
`L_1 ... L_K`, configured non-negative weights are right-aligned to the sampled
pass count and normalized to sum to one:

```text
L = sum_k w_k L_k
```

Examples:

```yaml
pass_loss_weights: [0.5, 0.5]
pass_loss_weights: [0.1, 0.9]
pass_loss_weights: [0.05, 0.20, 0.75]
pass_loss_weights: [0.0, 1.0]
```

The last form is useful for a wiring-only Phase A. `null` means uniform weights.
The objective machinery is common to FBT, MemoryAdd, and MemoryTape32 so architecture comparisons do
not silently change supervision.

## Pass-count schedule

Pass-count sampling and loss weighting are independent. `pass_schedule` is a
stateful token-indexed list of stages. Each stage supplies a probability mass
over positive pass counts; the final stage is unbounded.

```yaml
pass_schedule:
  - until_tokens: 2000000
    probabilities:
      2: 1.0
  - probabilities:
      1: 0.50
      2: 0.45
      3: 0.05
```

The scheduler has an independent RNG, sample counter, and pass histogram. Its
state is checkpointed so interrupted/resumed training reproduces the same
future pass-count draws.

The canonical Mac wiring configs currently use fixed `K=2`. Mixed schedules
remain an experimental knob, not a requirement or an implementation limit.

## Phase A: wiring

Phase A exists only for multipass variants.

- all pretrained TinyMistral parameters are frozen;
- only `added_parameters()` are trainable;
- every configured pass count must be at least 2;
- pass 1 is computed without retaining an autograd graph because it contains no
  trainable Phase-A parameters;
- later passes remain differentiable through the added recurrent pathway.

The checked-in memory Phase-A configs use fixed two-pass batches, final-pass-only
supervision, and a 1,048,576-token frozen-backbone budget. This is the mature
wiring recipe established by the completed campaign, not a claim that K=2 is
architecturally special.

## Phase B: adaptation

Phase B makes all model parameters differentiable, but the optimizer uses two
parameter groups:

```text
pretrained parameters
architecture-added parameters
```

Their base LRs are independently configurable:

```yaml
pretrained_learning_rate: 1.0e-7
added_learning_rate: 1.0e-6
```

A zero or very small pretrained LR is therefore possible without changing the
phase semantics. The canonical memory Phase-B configs use the selected `1e-7`
pretrained LR and `1e-6` added LR from the controlled dose-response campaign.

## LR schedules

The LR schedule supplies a common multiplicative factor to each parameter
group's base LR. Supported schedules are:

- `constant`
- `cosine`
- `piecewise_linear`

If `lr_schedule` is omitted, the original bootstrap cosine fields
`warmup_tokens` and `min_lr_ratio` retain their previous behavior.

## `init_from` and `resume_from`

These concepts are deliberately distinct.

### `resume_from`

Exact continuation of one trajectory. It restores:

- model state;
- optimizer state and parameter groups;
- shuffled data sampler position/permutation/RNG;
- pass-scheduler RNG, sample count, and histogram;
- Python/PyTorch/CUDA RNG state when applicable;
- optimizer/micro-step counters;
- unique-token and token-equivalent compute counters;
- phase and resolved experiment config.

### `init_from`

Loads only model parameters from another experiment checkpoint. Optimizer,
sampler, schedules, RNG streams, and token counters start fresh. This is the
intended Phase-A -> Phase-B boundary.

Phase-A compute should therefore be reported separately rather than silently
rolled into the Phase-B token budget.

## Token accounting

For each microbatch with `N` observed token IDs and `K` backbone passes:

```text
unique_tokens_seen += N
token_equivalent_compute += N * K
```

These are deliberately distinct. A mixed-pass schedule changes compute without
changing the number of unique training tokens.

## Evaluation

### One-pass held-out NLL

`scripts/eval_nll.py` retains the original one-pass model interface and is the
control metric.

### Pass-depth NLL

`scripts/eval_pass_depth.py` explicitly iterates a multipass model to a chosen
pass depth on the fixed validation artifact. It records:

- NLL and perplexity for every pass;
- per-source NLL for every pass;
- RMS difference between successive final top-layer hidden states.

This is the principal wiring/stability diagnostic.

### Previous-state interventions

`scripts/eval_memory_interventions.py` supports `memory_add` and
`memory_tape32`. It compares the real previous-pass state against an all-zero
state and a state taken from a different validation sequence. For MemoryAdd,
zero state is an exact vanilla input path even after training, because
`RMSNorm(0)=0` and the memory projection has no bias. The evaluator additionally
reports MemoryAdd residual RMS relative to token-embedding RMS to detect bypass.

### External benchmark harness

The existing lm-evaluation-harness adapter currently exercises ordinary
one-pass forward/generation semantics. It remains useful for the vanilla
control but is not yet the primary multipass metric.

### K-general incremental/recurrent evaluation

MemoryAdd and MemoryTape32 now support explicit cached inference with an
inference-time prompt depth `K >= 1`. This is separate from the pass-count
training schedule and from `eval_passes`.

`exact_incremental` keeps K independent TinyMistral self-attention KV streams.
At each appended position, every stream reads only the feedback memory that
existed before that position was processed. This snapshot-before-update rule is
required to preserve strict previous-position causality. The exact cached path
is tested against full finite-pass recomputation for K in `{1,2,3,4}`.

`recurrent` performs the same K-pass prompt prefill and then collapses to the
final pass's KV cache. For K>1 its initial feedback state is taken from pass
K-1, so the first processed continuation token is identical to exact K-pass
inference. Thereafter newly produced final-pass states are fed back to the same
stream. K=1 keeps feedback disabled and is exactly vanilla cached inference.

`scripts/eval_recurrent.py` teacher-forces a held-out continuation and reports
exact-K, recurrent-K, and pass-1 NLL by horizon, plus final-hidden RMS drift and
cosine similarity. The current K=2-trained checkpoints should be evaluated
first, but the implementation accepts arbitrary positive prefill K without a
code change.

## Generation boundary

The recurrent **state transition and teacher-forced evaluator** are implemented
for MemoryAdd and MemoryTape32. Free-running sampling is still deliberately
separate: public multipass `generate()` continues to delegate to vanilla
TinyMistral so lm-eval and ordinary model-call semantics do not change silently.
FBT does not yet opt into the cached recurrent hooks.

## Current Mac operating point

The first useful run is intentionally modest:

```text
FBT Phase A             fixed K=2
FBT Phase B             fixed K=2
MemoryAdd Phase A       fixed K=2
MemoryAdd Phase B       fixed K=2 (only after Phase-A gate)
MemoryTape32 Phase A    fixed K=2
MemoryTape32 Phase B    fixed K=2
```

For each multipass model inspect:

- pass-1 NLL;
- pass-2 NLL before/after wiring;
- gradient norms and finiteness;
- pass-depth NLL through at least 4-8 passes;
- hidden-state delta RMS;
- added parameter count;
- tokens/sec and token-equivalent compute.

The finite-pass wiring and conservative Phase-B gates have now been cleared for
MemoryAdd and MemoryTape32. Detailed campaign results and exact sweep configs live
under `experiments/memory_phase_b/`. Their next research gate is the exact-K versus
collapsed-recurrent continuation experiment described above. K=3/multi-depth
training should remain a response to measured recurrent drift rather than be
introduced before that drift is characterized.
