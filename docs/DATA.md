# Dolmino data artifact

The training runner does **not** live-stream remote data. A one-time preparation
step converts a small, pinned slice of `allenai/dolmino-mix-1124` into an exact
local token artifact. Every later architecture will therefore see the same token
IDs in the same block order.

## Published 50B recipe

The upstream percentages are rounded and sum to 100.01%, so the materializer
normalizes them before largest-remainder block allocation.

| Source config | Published mix % |
| --- | ---: |
| `dclm` | 47.20 |
| `flan` | 16.60 |
| `pes2o` | 5.85 |
| `wiki` | 7.11 |
| `stackexchange` | 2.45 |
| `math` | 20.80 |

A block belongs to exactly one source. At sequence length `T`, the requested
source token yield is therefore exact to one `T`-token block. At realistic
budgets the rounding error is negligible and is recorded in `manifest.json`.

## Split construction

For every source, one deterministic shuffled streaming iterator is created.
Validation documents are consumed first; their final partially used document is
discarded. Training then starts from the next document. This prevents a source
document from being shared between the local validation and training artifacts.

Documents are tokenized with the pinned TinyMistral `tokenizer.json`. A BOS token
is inserted as an explicit separator before every source document. Blocks are
fixed length, unpadded, and carry no attention mask so the vanilla MPS/CUDA fast
attention paths remain eligible.

## On-disk format

```text
artifact/
  train.bin                 uint16 [num_train_blocks, sequence_length]
  train.sources.bin         uint8  [num_train_blocks]
  validation.bin            uint16 [num_validation_blocks, sequence_length]
  validation.sources.bin    uint8  [num_validation_blocks]
  manifest.json
```

TinyMistral's 32,005-token vocabulary fits in `uint16`. The manifest contains
source allocations, tokenizer hash, requested/resolved dataset revisions,
recipe name, streaming shuffle buffer, preparation seed, and SHA-256 hashes of
every binary artifact. Dependency versions are subsequently frozen by the
project `uv.lock` generated on the machine that first resolves the online
dependencies.
