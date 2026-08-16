from __future__ import annotations

from dataclasses import dataclass


DOLMINO_REPO_ID = "allenai/dolmino-mix-1124"
# Current public dataset revision observed while this bootstrap was authored.
# The preparation code still resolves and records the actual requested revision.
DOLMINO_REFERENCE_REVISION = "1c2f43706986135c6799d9917e0d06ecef7fb1bb"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    config_name: str
    weight: float


DOLMINO_50B_SOURCES = (
    SourceSpec("dclm", "dclm", 0.4720),
    SourceSpec("flan", "flan", 0.1660),
    SourceSpec("pes2o", "pes2o", 0.0585),
    SourceSpec("wiki", "wiki", 0.0711),
    SourceSpec("stackexchange", "stackexchange", 0.0245),
    SourceSpec("math", "math", 0.2080),
)


def normalized_weights(sources: tuple[SourceSpec, ...] = DOLMINO_50B_SOURCES) -> tuple[float, ...]:
    total = sum(item.weight for item in sources)
    if total <= 0:
        raise ValueError("mixture weights must sum to a positive value")
    return tuple(item.weight / total for item in sources)


def allocate_blocks(total_blocks: int, sources: tuple[SourceSpec, ...] = DOLMINO_50B_SOURCES) -> dict[str, int]:
    """Largest-remainder allocation, preserving every source in small artifacts."""
    if total_blocks < 0:
        raise ValueError("total_blocks must be non-negative")
    if total_blocks and total_blocks < len(sources):
        raise ValueError("too few blocks to represent every mixture source")
    if total_blocks == 0:
        return {source.name: 0 for source in sources}
    weights = normalized_weights(sources)
    exact = [total_blocks * weight for weight in weights]
    counts = [int(value) for value in exact]
    remaining = total_blocks - sum(counts)
    order = sorted(
        range(len(sources)),
        key=lambda i: (exact[i] - counts[i], -i),
        reverse=True,
    )
    for i in order[:remaining]:
        counts[i] += 1

    # A very small dev split can round a low-weight source to zero. Preserve
    # coverage by transferring one block from the most overrepresented donor.
    zero_indices = [i for i, count in enumerate(counts) if count == 0]
    for zero in zero_indices:
        donors = [i for i, count in enumerate(counts) if count > 1]
        if not donors:
            raise ValueError("cannot preserve every source with this block budget")
        donor = max(donors, key=lambda i: counts[i] - exact[i])
        counts[donor] -= 1
        counts[zero] = 1
    return {source.name: count for source, count in zip(sources, counts)}
