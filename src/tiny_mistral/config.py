from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MistralConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096 * 32
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    sliding_window: int | None = 4096
    attention_dropout: float = 0.0
    torch_dtype: str | None = None
    model_type: str = "mistral"
    architectures: tuple[str, ...] = ("MistralForCausalLM",)
    transformers_version: str | None = None
    name_or_path: str | None = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        self.validate()

    def validate(self) -> None:
        if self.model_type != "mistral":
            raise ValueError(f"expected model_type='mistral', got {self.model_type!r}")
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("hidden/intermediate sizes must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim is None or self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError(
                "this reference expects num_attention_heads * head_dim == hidden_size"
            )
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive or None")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.hidden_act != "silu":
            raise ValueError("this austere reference currently supports hidden_act='silu' only")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MistralConfig":
        valid = {f.name for f in fields(cls)}
        mapped = dict(raw)
        if "_name_or_path" in mapped and "name_or_path" not in mapped:
            mapped["name_or_path"] = mapped.pop("_name_or_path")
        if "architectures" in mapped and isinstance(mapped["architectures"], list):
            mapped["architectures"] = tuple(mapped["architectures"])
        kwargs = {k: v for k, v in mapped.items() if k in valid}
        return cls(**kwargs)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "MistralConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["_name_or_path"] = data.pop("name_or_path")
        data["architectures"] = list(data["architectures"])
        return data

    def to_json_file(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")


def tiny_mistral_248m_config() -> MistralConfig:
    """Exact architecture fields from M4-ai/TinyMistral-248M-v3 config.json."""
    return MistralConfig(
        vocab_size=32005,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=32,
        attention_dropout=0.0,
        torch_dtype="bfloat16",
        transformers_version="4.45.2",
        name_or_path="M4-ai/TinyMistral-248M-v3",
    )
