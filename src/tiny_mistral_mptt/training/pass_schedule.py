from __future__ import annotations

import random
from collections import Counter
from typing import Any

from ..config import normalize_pass_schedule


class PassScheduler:
    """Stateful token-indexed sampler for the number of passes per microbatch."""

    def __init__(self, stages: list[dict[str, Any]] | None, *, seed: int):
        self.stages = normalize_pass_schedule(stages)
        self._rng = random.Random(int(seed))
        self.samples = 0
        self.histogram: Counter[int] = Counter()

    def _stage(self, tokens_seen: int) -> dict[str, Any]:
        if tokens_seen < 0:
            raise ValueError("tokens_seen must be non-negative")
        for stage in self.stages:
            until = stage["until_tokens"]
            if until is None or tokens_seen < until:
                return stage
        raise RuntimeError("pass schedule has no active stage")

    def sample(self, tokens_seen: int) -> int:
        probabilities = self._stage(tokens_seen)["probabilities"]
        draw = self._rng.random()
        cumulative = 0.0
        selected = max(probabilities)
        for passes, probability in probabilities.items():
            cumulative += probability
            if draw < cumulative:
                selected = int(passes)
                break
        self.samples += 1
        self.histogram[selected] += 1
        return selected

    def state_dict(self) -> dict[str, Any]:
        return {
            "stages": self.stages,
            "rng_state": self._rng.getstate(),
            "samples": self.samples,
            "histogram": dict(self.histogram),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        recorded_stages = normalize_pass_schedule(state["stages"])
        if recorded_stages != self.stages:
            raise ValueError("pass schedule changed across resume")
        self._rng.setstate(state["rng_state"])
        self.samples = int(state["samples"])
        self.histogram = Counter(
            {int(key): int(value) for key, value in state["histogram"].items()}
        )
