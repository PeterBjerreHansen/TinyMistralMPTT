from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    variant: str = "vanilla"
    model_dir: str = "checkpoints/TinyMistral-248M-v3"
    data_dir: str = "data/dolmino/dev_512"
    output_dir: str = "runs/vanilla-dev"
    device: str = "auto"
    dtype: str = "float32"
    attention_backend: str = "auto"
    seed: int = 1337
    batch_size: int = 1
    grad_accum_steps: int = 1
    max_unique_tokens: int = 65_536
    learning_rate: float = 1e-6
    min_lr_ratio: float = 0.1
    warmup_tokens: int = 0
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every_tokens: int = 32_768
    eval_batches: int = 16
    checkpoint_every_tokens: int = 65_536
    resume_from: str | None = None

    def validate(self) -> None:
        if self.variant != "vanilla":
            raise ValueError(
                f"this bootstrap supports only variant='vanilla'; got {self.variant!r}. "
                "FBT/memory variants are intentionally deferred to the next phase."
            )
        if self.batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch_size and grad_accum_steps must be positive")
        if self.max_unique_tokens <= 0:
            raise ValueError("max_unique_tokens must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.warmup_tokens < 0:
            raise ValueError("warmup_tokens must be non-negative")
        if self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("weight_decay must be non-negative and grad_clip positive")
        if self.eval_every_tokens < 0 or self.eval_batches < 0:
            raise ValueError("evaluation cadence/count must be non-negative")
        if self.checkpoint_every_tokens < 0:
            raise ValueError("checkpoint_every_tokens must be non-negative")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown experiment config fields: {unknown}")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("experiment config must be a YAML mapping")
    return ExperimentConfig.from_dict(raw)
