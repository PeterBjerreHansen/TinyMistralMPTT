# Substrate validation

## Question

Can the local TinyMistral implementation, data path, optimizer/checkpoint path,
and cached decoding be treated as a trustworthy substrate before adding memory
mechanisms?

## Record

The durable numerical and provenance gates are maintained in `docs/VALIDATION.md`
and `docs/UPSTREAMS.md`. Current reusable vanilla development configs live in
`configs/substrate/` and smoke configs in `configs/smoke/`.

Substrate validation is complete; future changes to `src/tiny_mistral/` must
re-clear the corresponding tests and source manifest.
