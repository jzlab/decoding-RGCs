"""DPS (Diffusion Posterior Sampling) backbone.

Vendored ADM / guided-diffusion U-Net plus the glue needed to drive it from the
existing Diffusers-shaped reconstruction loop. Selected with ``--backbone dps``.
"""

from .loader import (
    MODEL_PRESETS,
    DPSUNet,
    DPSUNetConfig,
    DPSUNetOutput,
    build_dps_scheduler,
    build_unet,
    epsilon_from_model_output,
    load_dps_unet,
)
from .unet import UNetModel

__all__ = [
    "MODEL_PRESETS",
    "DPSUNet",
    "DPSUNetConfig",
    "DPSUNetOutput",
    "UNetModel",
    "build_dps_scheduler",
    "build_unet",
    "epsilon_from_model_output",
    "load_dps_unet",
]
