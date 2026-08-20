from __future__ import annotations

import math

import torch

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM

from .multipass import HiddenRun, MultiPassVariant, shift_previous_hidden


class RecirculationVariant(MultiPassVariant):
    """Fixed-alpha source-layer recirculation.

    The output of ``source_layer`` is norm-matched and mixed into the output
    of the earlier ``destination_layer`` on later passes. The source is
    right-shifted before it is consumed, so position zero has no predecessor
    and cannot receive feedback from the current sequence.
    """

    variant_name = "recirculation"
    supports_cached_feedback = True

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        source_layer: int,
        destination_layer: int,
        alpha: float = 0.1,
    ):
        super().__init__(backbone)
        layer_count = len(backbone.model.layers)
        if not (0 <= destination_layer < source_layer < layer_count):
            raise ValueError(
                "require 0 <= destination_layer < source_layer < num_layers"
            )
        if not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be finite in [0,1]")
        self.source_layer = int(source_layer)
        self.destination_layer = int(destination_layer)
        self.alpha = float(alpha)

    @staticmethod
    def _norm_match(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        source_norm = torch.linalg.vector_norm(
            source.float(), ord=2, dim=-1, keepdim=True
        )
        destination_norm = torch.linalg.vector_norm(
            destination.float(), ord=2, dim=-1, keepdim=True
        )
        scale = (destination_norm / source_norm.clamp_min(1e-12)).to(source.dtype)
        return source * scale

    def _mix(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        *,
        valid_feedback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source.shape != destination.shape:
            raise ValueError("recirculation source and destination shapes differ")
        matched = self._norm_match(source, destination)
        candidate = self.alpha * matched + (1.0 - self.alpha) * destination
        if valid_feedback is None:
            return candidate
        return torch.where(valid_feedback[..., None], candidate, destination)

    @staticmethod
    def _cache_next_position(
        past_key_values: tuple[LayerKVCache, ...] | None,
    ) -> int:
        if not past_key_values:
            return 0
        positions = {cache.next_position for cache in past_key_values}
        if len(positions) != 1:
            raise ValueError("layer caches disagree on next position")
        return next(iter(positions))

    def _core(
        self,
        embeddings: torch.Tensor,
        *,
        recurrent_source: torch.Tensor | None,
        past_key_values: tuple[LayerKVCache, ...] | None,
        use_cache: bool,
        full_sequence_feedback: bool = False,
    ) -> HiddenRun:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must be [B,T,D]")
        if past_key_values is not None and len(past_key_values) != len(self.backbone.model.layers):
            raise ValueError("past_key_values must contain one cache per decoder layer")
        if recurrent_source is not None and recurrent_source.shape != embeddings.shape:
            raise ValueError("recirculation source and embeddings shapes differ")

        batch_size, seq_len, _ = embeddings.shape
        start = self._cache_next_position(past_key_values)
        position_ids = torch.arange(
            start,
            start + seq_len,
            device=embeddings.device,
            dtype=torch.long,
        )[None, :].expand(batch_size, -1)
        valid_feedback = None
        if recurrent_source is not None and full_sequence_feedback:
            valid_feedback = torch.ones(
                (batch_size, seq_len), dtype=torch.bool, device=embeddings.device
            )
            valid_feedback[:, 0] = False

        hidden_states = embeddings
        caches: list[LayerKVCache] | None = [] if use_cache else None
        source_capture: torch.Tensor | None = None
        for layer_index, layer in enumerate(self.backbone.model.layers):
            past = None if past_key_values is None else past_key_values[layer_index]
            hidden_states, cache = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past,
                use_cache=use_cache,
                fast_attention_compatible=past_key_values is None,
            )
            if caches is not None:
                if cache is None:
                    raise RuntimeError("recirculation layer did not return KV state")
                caches.append(cache)
            if layer_index == self.destination_layer and recurrent_source is not None:
                hidden_states = self._mix(
                    recurrent_source,
                    hidden_states,
                    valid_feedback=valid_feedback,
                )
            if layer_index == self.source_layer:
                source_capture = hidden_states

        if source_capture is None:
            raise RuntimeError("recirculation source layer was not reached")
        hidden_states = self.backbone.model.norm(hidden_states)
        return HiddenRun(
            hidden_states=hidden_states,
            feedback_source=source_capture,
            past_key_values=tuple(caches) if caches is not None else None,
        )

    def _run_first_state(self, input_ids: torch.Tensor) -> HiddenRun:
        return self._core(
            self.input_embeddings(input_ids),
            recurrent_source=None,
            past_key_values=None,
            use_cache=False,
        )

    def _run_feedback_state(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_source: torch.Tensor,
    ) -> HiddenRun:
        del input_ids
        return self._core(
            token_embeddings,
            recurrent_source=shift_previous_hidden(previous_source),
            past_key_values=None,
            use_cache=False,
            full_sequence_feedback=True,
        )

    def _run_first_state_cached(self, input_ids: torch.Tensor) -> HiddenRun:
        return self._core(
            self.input_embeddings(input_ids),
            recurrent_source=None,
            past_key_values=None,
            use_cache=True,
        )

    def _run_feedback_state_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_source: torch.Tensor,
    ) -> HiddenRun:
        del input_ids
        return self._core(
            token_embeddings,
            recurrent_source=shift_previous_hidden(previous_source),
            past_key_values=None,
            use_cache=True,
            full_sequence_feedback=True,
        )

    def _run_first_token_state_cached(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
    ) -> HiddenRun:
        return self._core(
            self.input_embeddings(input_ids),
            recurrent_source=None,
            past_key_values=past_key_values,
            use_cache=True,
        )

    def _run_feedback_token_state_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> HiddenRun:
        del token
        return self._core(
            token_embedding,
            recurrent_source=feedback_memory,
            past_key_values=past_key_values,
            use_cache=True,
        )

    # Legacy hidden-only hooks remain available to callers of the original
    # multipass API; the richer state hooks above carry the source layer.
    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        return self._run_feedback_state(input_ids, token_embeddings, previous_hidden).hidden_states

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        run = self._run_feedback_state_cached(input_ids, token_embeddings, previous_hidden)
        if run.past_key_values is None:
            raise RuntimeError("cached recirculation pass did not return KV state")
        return run.hidden_states, run.past_key_values

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        run = self._run_feedback_token_state_cached(
            token_embedding,
            feedback_memory,
            past_key_values,
            token=token,
        )
        if run.past_key_values is None:
            raise RuntimeError("cached recirculation token did not return KV state")
        return run.hidden_states, run.past_key_values

    def _feedback_memory_from_hidden(
        self,
        feedback_source: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids
        if feedback_source.ndim != 3 or feedback_source.shape[1] < 1:
            raise ValueError("feedback_source must be non-empty [B,T,D]")
        return feedback_source[:, -1:, :].detach()

    def _append_feedback_memory(
        self,
        feedback_memory: torch.Tensor,
        new_feedback_source: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> torch.Tensor:
        del token, position
        if feedback_memory.ndim != 3 or feedback_memory.shape[1] != 1:
            raise ValueError("recirculation feedback memory must be [B,1,D]")
        if new_feedback_source.ndim != 3 or new_feedback_source.shape[1] != 1:
            raise ValueError("new recirculation source must be [B,1,D]")
        return new_feedback_source.detach()
