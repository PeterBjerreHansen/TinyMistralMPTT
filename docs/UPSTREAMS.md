# Provenance and pinned external references

## Vanilla backbone

The vendored `src/tiny_mistral/` package is the vanilla TinyMistral reference
implementation derived from:

- repository: `PeterBjerreHansen/TinyMistralFork`
- validated model-source commit: `e44420d4190b6cfc1dc002c0ac67e364ef2f2de1`
- later validation-record commit: `b18a88d9c727e838a16fa39ed0e8d1e769b0e0c2`
- target checkpoint: `M4-ai/TinyMistral-248M-v3`
- checkpoint revision: `5afbc96ddc964c68282cd970ef49e8d1a5e81c52`
- Transformers architecture oracle: `4.45.2`

The research infrastructure must not modify vanilla behavior silently. The
checked-in baseline tests, real-checkpoint HF comparison scripts, and
`docs/VANILLA_SOURCE.sha256` are the guardrails for this bootstrap.

## Training data

- dataset: `allenai/dolmino-mix-1124`
- reference revision used by the checked-in data configs:
  `1c2f43706986135c6799d9917e0d06ecef7fb1bb`
- recipe: published 50B Stage-2 mixture token yields

No Dolmino data is redistributed by this repository. `prepare_data.py` streams
source documents only long enough to construct a requested local token artifact.
The generated manifest records the actually resolved dataset revision.

## Evaluation harness

- project: `EleutherAI/lm-evaluation-harness`
- package pin: `lm-eval==0.4.12`

The adapter is kept in this repository; the benchmark implementation and task
definitions remain upstream.

## Older research repositories

`PeterBjerreHansen/multi-pass-transformer-training` at
`79398be4ac33a7489029e6075bdce930a0ec44b2` and
`PeterBjerreHansen/chunked-mptt-core` at
`334c08e66104d327dc6aeb7e5b1fc564d23a7108` are design references for the
next model phase. No MPTT/MemoryTape model code from them is merged in this
bootstrap.

## Vanilla dispatch compatibility note

The research copy preserves the TinyMistral checkpoint hierarchy, attention
math, cache semantics, generation semantics, and numerical model contract. It
contains one deliberate non-numerical hardening relative to model-source commit
`e44420d4`: `MistralDecoderLayer.forward` forwards the model-level
`fast_attention_compatible` Python boolean into `MistralAttention`. The upstream
commit already computes and threads that flag to the decoder layer, but does not
pass it through the final call. Forwarding it avoids repeating device-tensor
compatibility checks in every layer on the canonical unpadded training path; it
does not change which attention implementation is mathematically valid or alter
model parameters. This difference is recorded here rather than being presented
as a bit-for-bit source copy.
