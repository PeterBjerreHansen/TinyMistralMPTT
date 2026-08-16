from __future__ import annotations

import torch

from tiny_mistral.device import resolve_device, resolve_dtype


def test_resolve_cpu_and_dtypes():
    device = resolve_device("cpu")
    assert device.type == "cpu"
    assert resolve_dtype("auto", device) == torch.float32
    assert resolve_dtype("float32", device) == torch.float32
    assert resolve_dtype("float16", device) == torch.float16
    assert resolve_dtype("bfloat16", device) == torch.bfloat16


def test_mps_auto_dtype_is_float16_without_requiring_mps_hardware():
    assert resolve_dtype("auto", torch.device("mps")) == torch.float16
