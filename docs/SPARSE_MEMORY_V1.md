# Sparse-memory implementation branch (`sparse-memory-v1`)

This source snapshot is intended for a separate architecture branch cut from the
CUDA-batch-ready baseline. It adds two experimental variants without changing
the existing Vanilla, FBT, MemoryAdd, or MemoryTape32 scientific controls:

- `sparse_memory_tape`
- `memory_add_sparse_tape`

## Branch contract

The implementation deliberately keeps MemoryTape's reader fixed. Sparse tape
experiments manipulate only write selection and the learned identity-initialized
write layer. The hybrid is simply MemoryAdd plus that sparse tape.

The primary correctness gates are:

1. C=1 SparseMemoryTape exactly reproduces dense MemoryTape32 with matching
   backbone/reader weights before writer training.
2. A write at position `t` is not readable at `t`.
3. `memory_window` counts memory records.
4. cached exact inference matches full-prefix recomputation.
5. recurrent collapse uses the K-1 feedback state for the first continuation
   transition.
6. hybrid fast state updates every token while its tape changes only on writes.

## Experimental order

Do not treat the default `memory_write_stride: 8` as selected. It is an ergonomic
construction default only. First qualify write cadence at W=32, then window size,
then compare the selected sparse tape against MemoryAdd and the hybrid.

Token-triggered writes are implemented so the architecture does not need to be
rewritten later, but the repository intentionally does not yet alter the
TinyMistral tokenizer or Dolmino token stream with `<MEM>` insertions.
