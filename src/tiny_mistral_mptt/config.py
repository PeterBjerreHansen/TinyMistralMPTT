from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_VARIANTS = {"vanilla", "fbt", "memory_add", "memory_tape32"}
SUPPORTED_LR_SCHEDULES = {"constant", "cosine", "piecewise_linear"}


def _coerce_pass_probabilities(raw: Any) -> dict[int, float]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("pass-schedule probabilities must be a non-empty mapping")
    result: dict[int, float] = {}
    for key, value in raw.items():
        try:
            passes = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid pass count {key!r}") from exc
        probability = float(value)
        if passes < 1:
            raise ValueError("pass counts must be positive")
        if not math.isfinite(probability) or probability < 0:
            raise ValueError("pass probabilities must be finite and non-negative")
        result[passes] = probability
    total = sum(result.values())
    if total <= 0:
        raise ValueError("pass probabilities must contain positive mass")
    return {passes: probability / total for passes, probability in sorted(result.items())}


def normalize_pass_schedule(raw: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Validate and normalize a token-indexed pass-count schedule.

    Each stage has ``probabilities`` and may have an exclusive ``until_tokens``
    bound. The last stage must be unbounded. Example::

        [{"until_tokens": 1_000_000, "probabilities": {2: 1.0}},
         {"probabilities": {1: .5, 2: .45, 3: .05}}]
    """
    if raw is None:
        return [{"until_tokens": None, "probabilities": {1: 1.0}}]
    if not isinstance(raw, list) or not raw:
        raise ValueError("pass_schedule must be a non-empty list")
    stages: list[dict[str, Any]] = []
    previous_until = 0
    for index, stage in enumerate(raw):
        if not isinstance(stage, dict):
            raise ValueError("each pass_schedule stage must be a mapping")
        unknown = sorted(set(stage) - {"until_tokens", "probabilities"})
        if unknown:
            raise ValueError(f"unknown pass_schedule stage fields: {unknown}")
        until = stage.get("until_tokens")
        if until is not None:
            until = int(until)
            if until <= previous_until:
                raise ValueError("pass_schedule until_tokens must increase strictly")
            previous_until = until
        elif index != len(raw) - 1:
            raise ValueError("only the final pass_schedule stage may omit until_tokens")
        stages.append(
            {
                "until_tokens": until,
                "probabilities": _coerce_pass_probabilities(stage.get("probabilities")),
            }
        )
    if stages[-1]["until_tokens"] is not None:
        raise ValueError("the final pass_schedule stage must be unbounded")
    return stages


def validate_lr_schedule(raw: dict[str, Any] | None) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ValueError("lr_schedule must be a mapping")
    schedule_type = str(raw.get("type", "cosine"))
    if schedule_type not in SUPPORTED_LR_SCHEDULES:
        raise ValueError(f"unsupported lr_schedule type {schedule_type!r}")
    if schedule_type == "constant":
        unknown = sorted(set(raw) - {"type"})
        if unknown:
            raise ValueError(f"unknown constant lr_schedule fields: {unknown}")
        return
    if schedule_type == "cosine":
        unknown = sorted(set(raw) - {"type", "warmup_tokens", "min_multiplier"})
        if unknown:
            raise ValueError(f"unknown cosine lr_schedule fields: {unknown}")
        warmup = int(raw.get("warmup_tokens", 0))
        minimum = float(raw.get("min_multiplier", 0.1))
        if warmup < 0 or not 0 <= minimum <= 1:
            raise ValueError("cosine schedule requires warmup_tokens>=0 and min_multiplier in [0,1]")
        return
    unknown = sorted(set(raw) - {"type", "points"})
    if unknown:
        raise ValueError(f"unknown piecewise_linear lr_schedule fields: {unknown}")
    points = raw.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("piecewise_linear schedule requires non-empty points")
    last_token = -1
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("piecewise_linear points must be [tokens, multiplier] pairs")
        tokens, multiplier = int(point[0]), float(point[1])
        if tokens < 0 or tokens <= last_token:
            raise ValueError("piecewise_linear token coordinates must increase from >=0")
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError("piecewise_linear multipliers must be finite and non-negative")
        last_token = tokens


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
    architecture_seed: int = 4242
    batch_size: int = 1
    grad_accum_steps: int = 1
    max_unique_tokens: int = 65_536

    # ``learning_rate`` remains the backwards-compatible vanilla/base default.
    # Phase-B parameter groups may override it independently.
    learning_rate: float = 1e-6
    pretrained_learning_rate: float | None = None
    added_learning_rate: float | None = None
    min_lr_ratio: float = 0.1
    warmup_tokens: int = 0
    lr_schedule: dict[str, Any] | None = None

    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every_tokens: int = 32_768
    eval_batches: int = 16
    eval_passes: int = 1
    checkpoint_every_tokens: int = 65_536

    # Architecture/training protocol knobs.
    phase: str = "B"
    pass_schedule: list[dict[str, Any]] | None = None
    pass_loss_weights: list[float] | None = None
    memory_window: int = 32
    prefix_mixin_probability: float = 0.0

    # ``resume_from`` restores the exact run. ``init_from`` loads model weights
    # only and begins a fresh trajectory/optimizer/data schedule.
    resume_from: str | None = None
    init_from: str | None = None

    def normalized_pass_schedule(self) -> list[dict[str, Any]]:
        return normalize_pass_schedule(self.pass_schedule)

    @property
    def pretrained_lr(self) -> float:
        return self.learning_rate if self.pretrained_learning_rate is None else float(self.pretrained_learning_rate)

    @property
    def added_lr(self) -> float:
        return self.learning_rate if self.added_learning_rate is None else float(self.added_learning_rate)

    def validate(self) -> None:
        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(SUPPORTED_VARIANTS)}; got {self.variant!r}")
        if self.phase not in {"A", "B"}:
            raise ValueError("phase must be 'A' or 'B'")
        if self.resume_from and self.init_from:
            raise ValueError("resume_from and init_from are mutually exclusive")
        if self.batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch_size and grad_accum_steps must be positive")
        if self.max_unique_tokens <= 0:
            raise ValueError("max_unique_tokens must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for name, value in (
            ("pretrained_learning_rate", self.pretrained_learning_rate),
            ("added_learning_rate", self.added_learning_rate),
        ):
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.warmup_tokens < 0:
            raise ValueError("warmup_tokens must be non-negative")
        validate_lr_schedule(self.lr_schedule)
        if self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("weight_decay must be non-negative and grad_clip positive")
        if self.eval_every_tokens < 0 or self.eval_batches < 0:
            raise ValueError("evaluation cadence/count must be non-negative")
        if self.eval_passes < 1:
            raise ValueError("eval_passes must be positive")
        if self.checkpoint_every_tokens < 0:
            raise ValueError("checkpoint_every_tokens must be non-negative")
        if self.memory_window <= 0:
            raise ValueError("memory_window must be positive")
        if (
            not math.isfinite(float(self.prefix_mixin_probability))
            or not 0.0 <= float(self.prefix_mixin_probability) <= 1.0
        ):
            raise ValueError("prefix_mixin_probability must be finite and in [0, 1]")
        if self.variant != "fbt" and self.prefix_mixin_probability != 0.0:
            raise ValueError(
                "prefix_mixin_probability is currently supported only for variant=fbt"
            )
        schedule = self.normalized_pass_schedule()
        pass_counts = {passes for stage in schedule for passes in stage["probabilities"]}
        if self.variant == "vanilla" and pass_counts != {1}:
            raise ValueError("vanilla supports only one-pass training")
        if self.variant == "vanilla" and self.eval_passes != 1:
            raise ValueError("vanilla supports eval_passes=1 only")
        if self.phase == "A" and self.variant == "vanilla":
            raise ValueError("vanilla has no Phase-A parameters")
        if self.phase == "A" and any(passes < 2 for passes in pass_counts):
            raise ValueError("Phase A for multipass variants requires at least two passes on every batch")
        if self.pass_loss_weights is not None:
            if not self.pass_loss_weights:
                raise ValueError("pass_loss_weights must not be empty")
            weights = [float(value) for value in self.pass_loss_weights]
            if any(not math.isfinite(value) or value < 0 for value in weights):
                raise ValueError("pass_loss_weights must be finite and non-negative")
            if sum(weights) <= 0:
                raise ValueError("pass_loss_weights must contain positive mass")

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
