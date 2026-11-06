"""Build the DPS (ADM / guided-diffusion) backbone behind a Diffusers-shaped API.

The reconstruction loop in ``diffusion_reconstruct.py`` was written against
Diffusers' ``UNet2DModel``: it reads ``unet.config.sample_size`` and calls
``unet(sample, timestep).sample``. ``DPSUNet`` presents that same surface over
the vendored ADM network, so selecting a backbone does not change the sampler.

The one semantic difference the caller must honour is ``learn_sigma``. Both
published DPS checkpoints predict 6 channels for a 3-channel image: the first
half is epsilon, the second half interpolates the reverse-process variance
(``model_var_type: learned_range``). ``epsilon_from_model_output`` performs that
split; Diffusers' ``DDPMScheduler`` does the same internally when it is
configured with ``variance_type="learned_range"``, so ``scheduler.step`` should
still receive the full 6-channel tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn

from .unet import UNetModel

# Mirrors configs/model_config.yaml and configs/imagenet_model_config.yaml in
# the DPS repository. Keys match the reference's create_model signature.
MODEL_PRESETS: dict[str, dict] = {
    "imagenet256": {
        "image_size": 256,
        "num_channels": 256,
        "num_res_blocks": 2,
        "channel_mult": "",
        "learn_sigma": True,
        "class_cond": False,
        "use_checkpoint": False,
        "attention_resolutions": "32,16,8",
        "num_heads": 4,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "dropout": 0.0,
        "resblock_updown": True,
        "use_fp16": False,
        "use_new_attention_order": False,
    },
    "ffhq256": {
        "image_size": 256,
        "num_channels": 128,
        "num_res_blocks": 1,
        "channel_mult": "",
        "learn_sigma": True,
        "class_cond": False,
        "use_checkpoint": False,
        "attention_resolutions": "16",
        "num_heads": 4,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "dropout": 0.0,
        "resblock_updown": True,
        "use_fp16": False,
        "use_new_attention_order": False,
    },
}

NUM_CLASSES = 1000


@dataclass(frozen=True)
class DPSUNetConfig:
    """The subset of Diffusers' UNet config that the sampler reads."""

    sample_size: int
    in_channels: int
    out_channels: int
    learn_sigma: bool


class DPSUNetOutput(NamedTuple):
    """Mimics Diffusers' ``UNet2DOutput`` so call sites can use ``.sample``."""

    sample: torch.Tensor


def build_unet(preset: str | dict) -> tuple[UNetModel, DPSUNetConfig]:
    """Instantiate the ADM U-Net for a named preset or an explicit config dict.

    Args:
        preset (str | dict): A key of ``MODEL_PRESETS`` or a full config dict.

    Returns:
        tuple[UNetModel, DPSUNetConfig]: The network and its reduced config.
    """
    if isinstance(preset, str):
        if preset not in MODEL_PRESETS:
            raise KeyError(
                f"Unknown DPS model preset {preset!r}. "
                f"Available: {sorted(MODEL_PRESETS)}"
            )
        params = dict(MODEL_PRESETS[preset])
    else:
        params = dict(preset)

    image_size = params["image_size"]
    channel_mult = params.get("channel_mult", "")
    if channel_mult == "":
        # Reproduces create_model's per-resolution defaults.
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch) for ch in str(channel_mult).split(","))

    # attention_resolutions is given as a resolution (e.g. 16), but the network
    # compares against a downsampling factor, hence image_size // res.
    attention_resolutions = params.get("attention_resolutions", "16")
    attention_ds: list[int] = []
    if isinstance(attention_resolutions, int):
        attention_ds.append(image_size // attention_resolutions)
    elif isinstance(attention_resolutions, str):
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
    else:
        raise NotImplementedError(
            f"Unsupported attention_resolutions: {attention_resolutions!r}"
        )

    learn_sigma = bool(params.get("learn_sigma", False))
    out_channels = 6 if learn_sigma else 3

    model = UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=params["num_channels"],
        out_channels=out_channels,
        num_res_blocks=params["num_res_blocks"],
        attention_resolutions=tuple(attention_ds),
        dropout=params.get("dropout", 0.0),
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if params.get("class_cond", False) else None),
        use_checkpoint=params.get("use_checkpoint", False),
        use_fp16=params.get("use_fp16", False),
        num_heads=params.get("num_heads", 1),
        num_head_channels=params.get("num_head_channels", -1),
        num_heads_upsample=params.get("num_heads_upsample", -1),
        use_scale_shift_norm=params.get("use_scale_shift_norm", False),
        resblock_updown=params.get("resblock_updown", False),
        use_new_attention_order=params.get("use_new_attention_order", False),
    )
    config = DPSUNetConfig(
        sample_size=image_size,
        in_channels=3,
        out_channels=out_channels,
        learn_sigma=learn_sigma,
    )
    return model, config


class DPSUNet(nn.Module):
    """Diffusers-shaped wrapper around the ADM U-Net.

    Args:
        model (UNetModel): The vendored ADM network.
        config (DPSUNetConfig): Reduced config exposed as ``.config``.
    """

    def __init__(self, model: UNetModel, config: DPSUNetConfig):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, sample: torch.Tensor, timestep, **_: object) -> DPSUNetOutput:
        """Run the network, broadcasting a scalar timestep across the batch.

        Args:
            sample (torch.Tensor): Noisy state shaped (batch, 3, H, W).
            timestep: Scalar tensor, int, or per-sample 1-D tensor.

        Returns:
            DPSUNetOutput: ``.sample`` holds the raw network output, which has
            6 channels whenever ``learn_sigma`` is set.
        """
        timesteps = torch.as_tensor(timestep, device=sample.device)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(sample.shape[0])
        elif timesteps.shape[0] != sample.shape[0]:
            timesteps = timesteps.reshape(-1)[0].expand(sample.shape[0])
        return DPSUNetOutput(sample=self.model(sample, timesteps))


def epsilon_from_model_output(
    model_output: torch.Tensor, sample: torch.Tensor
) -> torch.Tensor:
    """Take the epsilon half of a possibly variance-learning model output.

    A ``learn_sigma`` network emits twice the image channels; the leading half
    is epsilon. A 3-channel output is returned unchanged, so this is safe to
    apply on both backbones.

    Args:
        model_output (torch.Tensor): Raw U-Net output.
        sample (torch.Tensor): The state it was produced from, for channel count.

    Returns:
        torch.Tensor: Epsilon prediction matching ``sample``'s channel count.
    """
    channels = sample.shape[1]
    if model_output.shape[1] == channels:
        return model_output
    if model_output.shape[1] == 2 * channels:
        return model_output[:, :channels]
    raise ValueError(
        f"Cannot interpret a {model_output.shape[1]}-channel model output for a "
        f"{channels}-channel sample."
    )


def load_dps_unet(
    checkpoint: str | Path,
    preset: str | dict = "imagenet256",
    device: str | torch.device = "cpu",
) -> DPSUNet:
    """Build the ADM U-Net and load a published DPS checkpoint into it.

    Unlike the reference ``create_model``, which catches every load error and
    silently falls back to random initialisation, this raises. A mis-specified
    architecture produces plausible-looking noise rather than an obvious
    failure, so the strict load is the main guard against that.

    Args:
        checkpoint (str | Path): Path to ``imagenet256.pt`` or ``ffhq_10m.pt``.
        preset (str | dict): Architecture preset name or explicit config dict.
        device (str | torch.device): Device to place the network on.

    Returns:
        DPSUNet: Loaded network in eval mode with gradients disabled.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"DPS checkpoint not found: {path}\n"
            "Download it from the link in the DPS README:\n"
            "  https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh\n"
            "for example:  gdown --folder 1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh -O models/"
        )

    model, config = build_unet(preset)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # strict=True catches both renamed/absent keys and shape disagreements; the
    # wrapper only adds which preset was assumed, since the raw message is long.
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        name = preset if isinstance(preset, str) else "<explicit config>"
        raise RuntimeError(
            f"Checkpoint {path.name} does not match the {name!r} architecture. "
            "Check that the checkpoint and the preset refer to the same model "
            f"(available presets: {sorted(MODEL_PRESETS)}).\n{error}"
        ) from error

    wrapper = DPSUNet(model, config).to(device).eval()
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    return wrapper


def build_dps_scheduler(num_train_timesteps: int = 1000) -> "DDPMScheduler":
    """Build the DDPM scheduler described by the DPS diffusion config.

    Mirrors ``configs/diffusion_config.yaml``: a linear beta schedule over 1000
    steps, epsilon parameterisation, ``learned_range`` variance and clipped
    denoised samples. The beta endpoints are guided-diffusion's values for a
    1000-step schedule, which coincide with Diffusers' ``"linear"`` defaults.

    Args:
        num_train_timesteps (int): Length of the training schedule.

    Returns:
        DDPMScheduler: Scheduler matching the reference sampler.
    """
    from diffusers import DDPMScheduler

    scale = 1000 / num_train_timesteps
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=scale * 0.0001,
        beta_end=scale * 0.02,
        beta_schedule="linear",
        variance_type="learned_range",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
    )
