import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.fbt import FBTVariant
from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant
from tiny_mistral_mptt.variants.memory_tape32 import MemoryTape32Variant


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS hardware/runtime unavailable",
)


@pytest.mark.parametrize("variant_name", ["fbt", "memory_add", "memory_tape32"])
def test_multipass_variants_forward_backward_on_mps(variant_name):
    config = micro_config()
    backbone = MistralForCausalLM(config, attention_backend="auto").to("mps", dtype=torch.float32)
    if variant_name == "fbt":
        model = FBTVariant(backbone, initialization_seed=17)
    elif variant_name == "memory_add":
        model = MemoryAddVariant(backbone)
    else:
        model = MemoryTape32Variant(backbone, memory_window=4, initialization_seed=17)
    model = model.to("mps", dtype=torch.float32)
    configure_phase(model, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="mps")
    output = model.compute_loss(ids, phase="A", passes=2, loss_weights=[0.0, 1.0])
    assert torch.isfinite(output.loss)
    output.loss.backward()
    grads = [parameter.grad for parameter in model.added_parameters() if parameter.grad is not None]
    assert grads
    assert all(bool(torch.isfinite(grad).all().item()) for grad in grads)
