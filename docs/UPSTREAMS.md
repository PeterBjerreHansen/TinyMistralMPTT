# Provenance and pinned external references

## Vanilla backbone

The vendored `src/tiny_mistral/` package is the vanilla TinyMistral reference
implementation derived from:

- repository: `PeterBjerreHansen/TinyMistralFork`
- validated model-source commit: `e44420d4190b6cfc1dc002c0ac67e364ef2f2de1`
- target checkpoint: `M4-ai/TinyMistral-248M-v3`
- checkpoint revision: `5afbc96ddc964c68282cd970ef49e8d1a5e81c52`
- Transformers architecture oracle: `4.45.2`

The research infrastructure must not modify vanilla behavior silently. Baseline
tests, real-checkpoint HF comparison scripts, and `docs/VANILLA_SOURCE.sha256`
are the guardrails.

## FBT architecture reference

The `fbt` variant is based on the asymmetric latent-feedback construction in:

- Xi Wang et al., *Full-bandwidth Transformer*, arXiv:2608.08888

This repository does **not** claim to reproduce the paper's full model or
pretraining recipe. TinyMistral is retrofitted after pretraining, and pass-loss,
pass-count, Phase-A, and Phase-B schedules remain explicitly configurable.

## Earlier MPTT research reference

`PeterBjerreHansen/multi-pass-transformer-training` at
`79398be4ac33a7489029e6075bdce930a0ec44b2` is a design reference for:

- previous-pass top-layer memory semantics;
- strict one-token shift / recurrent causality;
- the MemoryAdd idea of injecting previous-pass state as an additive input
  residual;
- per-layer MemoryTape cross-attention;
- flexible right-aligned pass-loss weighting.

The current `memory_add` control deliberately differs from the older standalone
`MemoryAddTransformer` in one respect: it reuses the previous final top-layer
hidden state directly instead of learning a separate memory-write projection.
That choice keeps its recurrent bandwidth aligned with FBT and isolates the
reader/fusion mechanism for the TinyMistral retrofit experiment.

The TinyMistral implementations are written against the new repository's own
Mistral/GQA/local-attention interfaces rather than importing the old package.

## Explicit current non-dependency

`PeterBjerreHansen/chunked-mptt-core` is **not part of the current model phase**.
No chunked-memory layout, memory token, writer, cache, or hybrid semantics should
be inferred from the present FBT/MemoryAdd/MemoryTape32 implementation.

## Training data

- dataset: `allenai/dolmino-mix-1124`
- reference revision used by checked-in data configs:
  `1c2f43706986135c6799d9917e0d06ecef7fb1bb`
- recipe: published 50B Stage-2 mixture token yields

No Dolmino data is redistributed by this repository. `prepare_data.py` streams
source documents only long enough to construct the requested local token
artifact. The generated manifest records the resolved dataset revision.

## Evaluation harness

- project: `EleutherAI/lm-evaluation-harness`
- package pin: `lm-eval==0.4.12`

The adapter is kept in this repository; benchmark implementations and task
definitions remain upstream.

## Vanilla dispatch compatibility note

The research copy preserves the TinyMistral checkpoint hierarchy, attention
math, cache semantics, generation semantics, and numerical model contract. It
contains one deliberate non-numerical hardening relative to model-source commit
`e44420d4`: `MistralDecoderLayer.forward` forwards the model-level
`fast_attention_compatible` Python boolean into `MistralAttention`. The upstream
commit already computes and threads that flag to the decoder layer, but does not
pass it through the final call. Forwarding it avoids repeating device-tensor
compatibility checks in every layer on the canonical unpadded training path; it
does not alter model parameters or attention semantics.
