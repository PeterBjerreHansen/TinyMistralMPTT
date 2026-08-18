import pytest
import torch

from tiny_mistral_mptt.precision import PrecisionNotSupportedError, autocast_context


def test_autocast_none_is_portable():
    with autocast_context(torch.device("cpu"), None):
        value = torch.tensor([1.0])
    assert value.item() == 1.0


def test_cpu_bfloat16_autocast_is_rejected_by_training_contract():
    with pytest.raises(PrecisionNotSupportedError, match="only on CUDA or MPS"):
        autocast_context(torch.device("cpu"), "bfloat16")


def test_unavailable_mps_bfloat16_is_reported_as_unsupported():
    if torch.backends.mps.is_available():
        pytest.skip("this assertion targets hosts without MPS")
    with pytest.raises(PrecisionNotSupportedError, match="MPS.*unavailable"):
        autocast_context(torch.device("mps"), "bfloat16")


def test_unknown_autocast_dtype_is_rejected():
    with pytest.raises(ValueError, match="unsupported autocast dtype"):
        autocast_context(torch.device("cpu"), "float16")
