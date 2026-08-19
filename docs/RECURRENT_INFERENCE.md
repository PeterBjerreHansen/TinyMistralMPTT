# K-general incremental and recurrent inference

This document defines the cached inference contract for `memory_add` and
`memory_tape32`. The implementation is intentionally independent of the current
K=2 training protocol: prompt refinement depth is an inference-time
hyperparameter `K >= 1`.

## Two modes

### Exact incremental K

`exact_incremental` maintains K independent TinyMistral self-attention KV
streams. Stream 1 is ordinary vanilla TinyMistral. For MemoryAdd, stream
`k > 1` consumes

```text
x_t^k = e_t + W_M RMSNorm(h_(t-1)^(k-1)).
```

For MemoryTape32, stream `k > 1` reads the retained strict-past tail from stream
`k-1`, up to `memory_window` top-layer states. Each appended token costs K
backbone token steps, but no prefix is recomputed.

The key causality invariant is **snapshot before update**. All stream-k
computations at position `t` read the stream-(k-1) feedback memory that existed
before position `t`. Newly computed `h_t` states are appended only after all K
streams finish the token. This prevents same-position leakage.

The exact cached path is an oracle implementation: on CPU/reference attention,
its final-token hidden states and logits are tested against full-prefix
`compute_passes(..., passes=K)` recomputation for K in `{1,2,3,4}`, including
sequences longer than the self-attention sliding window.

### Collapsed recurrent K

`recurrent` uses the same K-pass prompt prefill, then retains only:

- the final pass-K TinyMistral KV cache; and
- for K>1, feedback memory from pass K-1.

For MemoryAdd the feedback memory is one `[B,1,D]` top-layer state. For
MemoryTape32 it is an oldest-to-newest ring of at most `memory_window` states.

This makes the first processed continuation token exact: it sees the same
pass-(K-1) feedback source and pass-K history as exact incremental inference.
After that token, the newly produced pass-K state is written into the feedback
memory and the model closes its own recurrent loop. The approximation therefore
begins only on the subsequent feedback transition.

K=1 is a strict boundary case. Feedback is disabled, so both exact and recurrent
modes reduce to ordinary cached vanilla TinyMistral.

## Public API

The low-level API is explicit and does not alter ordinary model calls:

```python
from tiny_mistral_mptt.inference import prefill, decode_step

state = prefill(model, input_ids, passes=K, mode="exact_incremental")
state = decode_step(model, state, observed_token)

state = prefill(model, input_ids, passes=K, mode="recurrent")
state = decode_step(model, state, observed_token)
```

Dedicated helpers `prefill_exact`, `prefill_recurrent`, `exact_decode_step`, and
`recurrent_decode_step` are also exported. State objects are immutable
containers; decode steps return new state objects rather than mutating prior
state metadata or memory tensors.

Public `MultiPassVariant.generate()` remains the vanilla generator on purpose.
Free-running recurrent sampling should be added only after teacher-forced drift
has been measured and the explicit inference path is trusted on the target
hardware.

MemoryAdd and MemoryTape32 explicitly opt into cached feedback. FBT currently
supports only the K=1 vanilla cached boundary; requesting K>1 through the cached
inference API raises a clear capability error.

## Teacher-forced evaluator

`scripts/evaluate_recurrent_inference.py` compares three modes on the same
held-out suffix:

1. exact incremental K;
2. collapsed recurrent K;
3. pass-1/vanilla cached inference from the same checkpoint.

Example:

```bash
uv run python scripts/evaluate_recurrent_inference.py \
  --config <experiment-config.yaml> \
  --checkpoint <checkpoint.pt> \
  --prefill-passes 1 2 3 4 8 \
  --prompt-tokens 256 \
  --continuation-tokens 256 \
  --horizons 1 2 4 8 16 32 64 128 256 \
  --output /tmp/recurrent_inference.json
```

`--prefill-passes` accepts one or more positive K values. It is deliberately not
stored in the training `ExperimentConfig`: the checkpoint describes how weights
were trained, while prefill depth describes how those weights are exercised at
inference.

For each K the JSON output includes cumulative NLL at requested horizons,
per-offset NLL, recurrent-minus-exact and recurrent-minus-vanilla gaps, and
final-hidden RMS/cosine drift after each processed suffix token.

For a 256-token prompt and 256-token suffix, `prefill_passes` may be swept
independently of training K. Training depth and inference depth are separate
experimental variables; the experiment record must state both.

## MemoryTape32 cached reader

Full finite-pass MemoryTape32 uses `strict_past_local_attention`, where query and
previous-pass memory sequences are aligned and the local strict-past mask is
constructed explicitly. Cached decoding instead supplies a memory bank that is
already known to be strictly earlier than the one-token query. The dedicated
`memory_bank_attention` primitive therefore performs GQA attention directly
against that bank without adding a same-position key.

The recurrent ring is capped at `memory_window`; the initial ring is the tail of
pass K-1 and, one token at a time, old prompt states are replaced by newly
produced recurrent states.

## Validation gates

Before interpreting recurrent NLL, the following must remain green:

- exact incremental equals full finite-pass recomputation for multiple K;
- K=1 exact and recurrent modes equal vanilla cached inference;
- K>1 recurrent prefill logits equal exact prefill logits;
- the first processed recurrent token equals the exact-K token step;
- MemoryAdd retains exactly one feedback vector;
- MemoryTape32 retains an ordered ring no longer than its memory window;
- no same-position source state is exposed during exact K-stream updates;
- cached absolute positions remain correct beyond the TinyMistral SWA window;
- CPU/reference tests pass and the checked-in MPS smoke tests pass on Apple
  hardware.
