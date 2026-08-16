from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


try:  # Optional dependency: imported only for the external benchmark battery.
    from lm_eval import utils as lm_eval_utils
    from lm_eval.api.model import TemplateLM
except ImportError:  # pragma: no cover - environment dependent
    TemplateLM = None  # type: ignore[assignment]
    lm_eval_utils = None  # type: ignore[assignment]




@torch.no_grad()
def score_token_continuation(
    model,
    *,
    device: str | torch.device,
    max_length: int,
    context_enc: list[int],
    continuation_enc: list[int],
) -> tuple[float, bool]:
    """Score one already-tokenized causal context/continuation pair.

    This is intentionally independent of lm-eval so the indexing contract can
    be unit-tested even when the optional harness package is unavailable.
    """
    if not context_enc:
        raise ValueError("context_enc must contain at least one prefix/context token")
    if not continuation_enc:
        return 0.0, True
    combined = list(context_enc) + list(continuation_enc)
    if len(combined) > max_length + 1:
        removed = len(combined) - (max_length + 1)
        combined = combined[removed:]
        remaining_context = max(len(context_enc) - removed, 0)
    else:
        remaining_context = len(context_enc)
    if len(combined) < 2:
        return 0.0, True
    input_tokens = combined[:-1]
    target_tokens = combined[1:]
    start = max(remaining_context - 1, 0)
    scored_targets = target_tokens[start:]
    if not scored_targets:
        return 0.0, True
    ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    output = model(ids, use_cache=False)
    logits = output.logits[:, start : start + len(scored_targets), :].float()
    targets = torch.tensor([scored_targets], dtype=torch.long, device=device)
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    total = float(chosen.sum().cpu())
    greedy = bool(torch.equal(logits.argmax(dim=-1), targets))
    return total, greedy


class _TokenizerFacade:
    def __init__(self, path: str | Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("evaluation requires: uv sync --extra eval") from exc
        self.inner = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return self.inner.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.inner.decode(ids, skip_special_tokens=False)


def _build_lm_eval_class():
    if TemplateLM is None:  # pragma: no cover - optional dependency
        return None

    class TinyMistralHarnessLM(TemplateLM):
        """Minimal single-process lm-evaluation-harness adapter for this repo."""

        backend = "causal"

        def __init__(
            self,
            model,
            *,
            tokenizer_path: str | Path,
            device: str | torch.device,
            max_gen_toks: int = 128,
        ):
            super().__init__()
            self.model = model
            self._device = torch.device(device)
            self._tokenizer_facade = _TokenizerFacade(tokenizer_path)
            self._max_gen_toks = int(max_gen_toks)
            self.batch_size = 1
            self.model.eval()

        @property
        def eot_token_id(self) -> int:
            return int(self.model.config.eos_token_id)

        @property
        def prefix_token_id(self) -> int:
            return int(self.model.config.bos_token_id)

        @property
        def max_length(self) -> int:
            return int(self.model.config.max_position_embeddings)

        @property
        def max_gen_toks(self) -> int:
            return self._max_gen_toks

        @property
        def tokenizer_name(self) -> str:
            return "TinyMistral-248M-v3-tokenizer"

        def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
            # TinyMistral's baseline experiments use raw tokenizer.json encoding;
            # BOS is inserted explicitly only when the harness needs an empty context.
            return self._tokenizer_facade.encode(string)

        def tok_decode(self, tokens: list[int]) -> str:
            return self._tokenizer_facade.decode([int(token) for token in tokens])

        @torch.no_grad()
        def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **kwargs):
            results: list[tuple[float, bool]] = []
            for cache_key, context_enc, continuation_enc in requests:
                answer = score_token_continuation(
                    self.model,
                    device=self._device,
                    max_length=self.max_length,
                    context_enc=list(context_enc),
                    continuation_enc=list(continuation_enc),
                )
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
            return results

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
            if lm_eval_utils is None:
                raise RuntimeError("lm-eval is unavailable")
            results: list[float] = []
            for request in requests:
                string = request.args[0]
                tokens = self.tok_encode(string)
                windows = list(
                    map(
                        lm_eval_utils.make_disjoint_window,
                        lm_eval_utils.get_rolling_token_windows(
                            token_list=tokens,
                            prefix_token=self.prefix_token_id,
                            max_seq_len=self.max_length,
                            context_len=1,
                        ),
                    )
                )
                token_requests = [(None,) + window for window in windows]
                scored = self._loglikelihood_tokens(token_requests, disable_tqdm=disable_tqdm)
                total = sum(item[0] for item in scored)
                results.append(total)
                self.cache_hook.add_partial("loglikelihood_rolling", (string,), total)
            return results

        @torch.no_grad()
        def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
            results: list[str] = []
            for request in requests:
                context, gen_kwargs = request.args
                gen_kwargs = dict(gen_kwargs or {})
                until = gen_kwargs.get("until", [])
                if isinstance(until, str):
                    until = [until]
                max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
                temperature = float(gen_kwargs.get("temperature", 0.0))
                top_k = gen_kwargs.get("top_k")
                context_ids = self.tok_encode(context)
                if not context_ids:
                    context_ids = [self.prefix_token_id]
                # Leave room for requested generation inside the vanilla RoPE limit.
                max_prompt = max(1, self.max_length - max_new)
                context_ids = context_ids[-max_prompt:]
                prompt = torch.tensor([context_ids], dtype=torch.long, device=self._device)
                generated = self.model.generate(
                    prompt,
                    max_new,
                    temperature=temperature,
                    top_k=None if top_k is None else int(top_k),
                )
                suffix_ids = generated[0, prompt.shape[1] :].tolist()
                text = self.tok_decode(suffix_ids)
                cut = len(text)
                for stop in until:
                    position = text.find(stop)
                    if position >= 0:
                        cut = min(cut, position)
                text = text[:cut]
                results.append(text)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
            return results

    return TinyMistralHarnessLM


TinyMistralHarnessLM = _build_lm_eval_class()


def make_lm_eval_adapter(model, *, tokenizer_path: str | Path, device: str | torch.device, max_gen_toks: int = 128):
    if TinyMistralHarnessLM is None:
        raise RuntimeError("lm-evaluation-harness is not installed; run: uv sync --extra eval")
    return TinyMistralHarnessLM(
        model,
        tokenizer_path=tokenizer_path,
        device=device,
        max_gen_toks=max_gen_toks,
    )
