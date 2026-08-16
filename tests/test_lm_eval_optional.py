import importlib.util
import pytest

from tiny_mistral_mptt.evaluation.lm_eval_adapter import TinyMistralHarnessLM


def test_lm_eval_is_cleanly_optional():
    available = importlib.util.find_spec("lm_eval") is not None
    if available:
        assert TinyMistralHarnessLM is not None
    else:
        assert TinyMistralHarnessLM is None
