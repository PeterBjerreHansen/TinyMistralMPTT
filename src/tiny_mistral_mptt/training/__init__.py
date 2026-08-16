from .checkpoint import TrainState, load_checkpoint, load_model_weights, save_checkpoint
from .phases import PhasePlan, configure_phase, vanilla_phase_a_is_noop

__all__ = [
    "PhasePlan",
    "TrainState",
    "Trainer",
    "configure_phase",
    "load_checkpoint",
    "load_model_weights",
    "save_checkpoint",
    "vanilla_phase_a_is_noop",
]


def __getattr__(name):
    # Avoid importing the trainer while variant modules import small training
    # utilities (loss weighting), which would create a package import cycle.
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    raise AttributeError(name)
