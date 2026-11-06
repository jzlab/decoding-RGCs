"""Alternate an unconditional DDPM prior with an RGC MAP likelihood update."""

from __future__ import annotations

# Running this file directly puts its own folder on sys.path, not the repository
# root, so the shared package would not resolve. Prepend the root before any
# first-party import. Harmless when the module is imported as part of a package.
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = str(_Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import argparse
from collections.abc import Sequence

import torch
from diffusers import DDPMScheduler, UNet2DModel

from diffusion.dps_backbone import build_dps_scheduler, epsilon_from_model_output, load_dps_unet
from PIL import Image
from tqdm.auto import tqdm

from diffusion.diffusion_map import DeterministicRateLikelihood
from diffusion.diffusion_retina import StaticRGCEncoder
from shared.sk_utils import Utils as u

def ddpm_rgb_to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    """Convert DDPM RGB samples from [-1, 1] to grayscale samples in [0, 1].

    Args:
        rgb (torch.Tensor): RGB DDPM samples shaped (batch, 3, height, width).

    Returns:
        torch.Tensor: Grayscale samples shaped (batch, 1, height, width).
    """

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("The selected DDPM prior must produce RGB tensors.")
    weights = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    luminance = (rgb * weights).sum(dim=1, keepdim=True)
    return ((luminance + 1.0) / 2.0).clamp(0, 1)


def _scale_guidance_gradient(
    gradient: torch.Tensor, guidance_scale: float, normalization: str
) -> tuple[torch.Tensor, float]:
    """Scale a likelihood gradient for a reverse-diffusion correction.

    Args:
        gradient (torch.Tensor): Gradient of the RGC loss with respect to x_t.
        guidance_scale (float): Requested correction magnitude.
        normalization (str): ``none`` for raw scaling or ``rms`` for RMS scaling.

    Returns:
        tuple[torch.Tensor, float]: Applied correction and unscaled gradient RMS.
    """
    gradient_rms = gradient.square().mean().sqrt()
    if normalization == "none":
        return guidance_scale * gradient, float(gradient_rms.detach().cpu())
    if normalization == "rms":
        return guidance_scale * gradient / gradient_rms.clamp_min(1e-12), float(gradient_rms.detach().cpu())
    raise ValueError(f"Unsupported guidance normalization: {normalization}")


class DiffusionMAPReconstructor:
    """Combine a response-agnostic DDPM prior with MAP rate guidance.

    Two priors are selectable. ``diffusers`` is the original path: a Diffusers
    ``UNet2DModel`` identified by ``model_id``. ``dps`` is the ADM /
    guided-diffusion network published with Diffusion Posterior Sampling,
    loaded from a local ``.pt`` checkpoint. The sampler itself is identical
    either way; only the prior and its scheduler change.

    Args:
        likelihood (DeterministicRateLikelihood): RGC likelihood used for guidance.
        model_id (str): Hugging Face identifier for the unconditional DDPM.
        device (str): Device used for model evaluation and reconstruction.
        backbone (str): ``diffusers`` for the Hugging Face prior, ``dps`` for the
            published DPS checkpoint.
        dps_checkpoint (str | None): Path to the DPS ``.pt``, required when
            ``backbone`` is ``dps``.
        dps_preset (str): Architecture preset naming the DPS checkpoint's shape.
    """

    def __init__(
        self,
        likelihood: DeterministicRateLikelihood,
        model_id: str = "google/ddpm-bedroom-256",
        device: str = "cpu",
        backbone: str = "diffusers",
        dps_checkpoint: str | None = None,
        dps_preset: str = "imagenet256",
    ):

        self.device = torch.device(device)
        self.backbone = backbone

        if backbone == "diffusers":
            self.unet = UNet2DModel.from_pretrained(model_id).to(self.device).eval()
            self.scheduler = DDPMScheduler.from_pretrained(model_id)
        elif backbone == "dps":
            if dps_checkpoint is None:
                raise ValueError(
                    "backbone='dps' requires --dps-checkpoint (or dps_checkpoint "
                    "in the diffusion params) pointing at the downloaded .pt file."
                )
            # The DPS checkpoints carry no scheduler, so it is reconstructed
            # from the reference diffusion config rather than loaded.
            self.unet = load_dps_unet(dps_checkpoint, dps_preset, self.device)
            self.scheduler = build_dps_scheduler()
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone!r}. Expected 'diffusers' or 'dps'."
            )

        self.likelihood = likelihood.to(self.device).eval()

        for parameter in self.unet.parameters():
            parameter.requires_grad_(False)
        for parameter in self.likelihood.parameters():
            parameter.requires_grad_(False)

    def _predict_original_sample(
        self, sample: torch.Tensor, noise_prediction: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Compute the DDPM clean-image estimate while retaining its input gradient.

        Args:
            sample (torch.Tensor): Current noisy DDPM state shaped (batch, 3, height, width).
            noise_prediction (torch.Tensor): U-Net prediction at the current timestep.
            timestep (torch.Tensor): Current discrete DDPM timestep.

        Returns:
            torch.Tensor: Predicted clean DDPM image shaped like ``sample``.
        """
        # A learn_sigma backbone (DPS) emits 6 channels: epsilon, then the
        # variance interpolation. Only the epsilon half enters the x0 estimate.
        # scheduler.step() still receives the full tensor and splits it itself.
        noise_prediction = epsilon_from_model_output(noise_prediction, sample)
        alpha_bar = self.scheduler.alphas_cumprod.to(sample.device)[timestep].to(sample.dtype)
        beta_bar = 1 - alpha_bar
        prediction_type = self.scheduler.config.prediction_type

        if prediction_type == "epsilon":
            predicted = (sample - beta_bar.sqrt() * noise_prediction) / alpha_bar.sqrt()
        elif prediction_type == "sample":
            predicted = noise_prediction
        elif prediction_type == "v_prediction":
            predicted = alpha_bar.sqrt() * sample - beta_bar.sqrt() * noise_prediction
        else:
            raise ValueError(f"Unsupported DDPM prediction type: {prediction_type}")

        if self.scheduler.config.thresholding:
            # The selected Google checkpoint does not use thresholding.  A
            # future checkpoint using it needs Diffusers' dynamic-thresholding
            # implementation here to exactly match scheduler.step().
            raise NotImplementedError("Dynamic thresholding is not implemented for likelihood guidance.")
        if self.scheduler.config.clip_sample:
            predicted = predicted.clamp(
                self.scheduler.config.clip_sample_range * -1,
                self.scheduler.config.clip_sample_range,
            )
        return predicted

    def reconstruct(
        self,
        observed_rates: Sequence[torch.Tensor],
        num_inference_steps: int = 250,
        guidance_scale: float = 0.05,
        guidance_normalization: str = "rms",
        seed: int | None = None,
    ) -> tuple[torch.Tensor, list[float]]:
        """Reconstruct one grayscale image by alternating prior and MAP updates.

        Args:
            observed_rates (Sequence[torch.Tensor]): Observed rates for each RGC mosaic.
            num_inference_steps (int): Number of reverse DDPM steps.
            guidance_scale (float): Magnitude of each likelihood-gradient correction.
            guidance_normalization (str): Raw or RMS-normalized likelihood-gradient scaling.
            seed (int | None): Optional random seed for the DDPM initial noise.

        Returns:
            tuple[torch.Tensor, list[float]]: Reconstructed grayscale image and loss trace.
        """

        if guidance_scale <= 0:
            raise ValueError("guidance_scale must be positive.")
        
        generator = torch.Generator(device=self.device)

        if seed is not None:
            generator.manual_seed(seed)
            
        sample_size = self.unet.config.sample_size
        sample = torch.randn((1, 3, sample_size, sample_size), device=self.device, generator=generator)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        losses: list[float] = []

        progress = tqdm(self.scheduler.timesteps, desc="Reverse diffusion", unit="step")
        for timestep in progress:
            # The U-Net weights are frozen, but its clean-image estimate remains
            # differentiable with respect to x_t. This is the required chain
            # rule for likelihood guidance, dL/dx_t = d x0/dx_t^T dL/dx0.
            sample = sample.detach().requires_grad_(True)
            noise_prediction = self.unet(sample, timestep).sample
            clean_rgb = self._predict_original_sample(sample, noise_prediction, timestep)
            clean_gray = ddpm_rgb_to_grayscale(clean_rgb)
            loss = self.likelihood.loss(clean_gray, observed_rates)
            (guidance_gradient,) = torch.autograd.grad(loss, sample)
            losses.append(float(loss.cpu()))
            correction, gradient_rms = _scale_guidance_gradient(
                guidance_gradient, guidance_scale, guidance_normalization
            )
            correction_rms = float(correction.square().mean().sqrt().detach().cpu())
            progress.set_postfix(
                rgc_loss=f"{losses[-1]:.5g}", grad_rms=f"{gradient_rms:.2e}", correction_rms=f"{correction_rms:.2e}"
            )

            # First take the discrete unconditional reverse-DDPM transition,
            # then apply its timestep-specific MAP likelihood correction.
            # ``guidance_gradient`` is dL/dx_t, not dL/dx0.
            with torch.no_grad():
                prior = self.scheduler.step(
                    noise_prediction.detach(), timestep, sample.detach(), generator=generator
                )
                sample = prior.prev_sample - correction

        return ddpm_rgb_to_grayscale(sample), losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a grayscale image from deterministic RGC rates.")
    parser.add_argument("--params", default=f"{_REPO_ROOT}/shared/params.yaml", help="Existing retina YAML configuration.")
    parser.add_argument("--diffusion-params", default=f"{_REPO_ROOT}/diffusion/diffusion_params.yaml",
                        help="Diffusion configuration containing an optional include_cell_types list.")
    parser.add_argument("--rates", required=True, help=".pt containing a list of (1, n_cells) rate tensors.")
    parser.add_argument("--output", default="diffusion_reconstruction.png",
                        help="Output grayscale image path, such as PNG or JPEG.")
    parser.add_argument("--loss-output", default=None,
                        help="Optional .pt path to save per-timestep likelihood losses.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--backbone", choices=("diffusers", "dps"), default=None,
                        help="Unconditional prior to guide. Defaults to the diffusers model.")
    parser.add_argument("--dps-checkpoint", default=None,
                        help="Path to the published DPS .pt checkpoint, for --backbone dps.")
    parser.add_argument("--dps-preset", default=None,
                        help="DPS architecture preset, such as imagenet256 or ffhq256.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None,
                        help="Spatial size used to retile the static hexagonal mosaics.")
    parser.add_argument("--include-cell-types", nargs="+", default=None,
                        help="Cell-type tags to use; overrides diffusion.include_cell_types in --params.")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--guidance-normalization", choices=("none", "rms"), default=None,
                        help="Use raw MAP gradients or normalize each correction to its RMS magnitude.")
    parser.add_argument("--rate-noise-std", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    diffusion_params = u.read_params(args.diffusion_params)
    args.model_id = args.model_id or diffusion_params["model_id"]
    args.backbone = args.backbone or diffusion_params.get("backbone", "diffusers")
    args.dps_checkpoint = args.dps_checkpoint or diffusion_params.get("dps_checkpoint")
    args.dps_preset = args.dps_preset or diffusion_params.get("dps_preset", "imagenet256")
    args.steps = args.steps or diffusion_params["num_inference_steps"]
    args.image_size = args.image_size or diffusion_params["image_size"]
    args.guidance_scale = args.guidance_scale or diffusion_params["guidance_scale"]
    args.guidance_normalization = args.guidance_normalization or diffusion_params.get("guidance_normalization", "rms")
    args.rate_noise_std = args.rate_noise_std or diffusion_params["rate_noise_std"]

    print(f"Loading observed RGC rates from {args.rates}...")
    print(f"Loading retina parameters from {args.params} and building {args.image_size}x{args.image_size} mosaics...")
    params = u.read_params(args.params)
    include_cell_types = args.include_cell_types or diffusion_params.get("include_cell_types")
    encoder = StaticRGCEncoder.from_params(
        params, (args.image_size, args.image_size), include_cell_types
    )
    saved_observations = torch.load(args.rates, map_location="cpu", weights_only=True)
    if isinstance(saved_observations, dict):
        observed_rates = saved_observations.get("rates")
        saved_cell_types = saved_observations.get("cell_types")
        if not isinstance(observed_rates, (list, tuple)) or not isinstance(saved_cell_types, list):
            raise TypeError("Rate metadata must contain a list of rates and a list of cell types.")
        if saved_cell_types != encoder.cell_type_names:
            raise ValueError(
                "Rate-file cell types do not match the selected encoder cell types. "
                f"Rate file: {saved_cell_types}; selected: {encoder.cell_type_names}."
            )
    elif isinstance(saved_observations, (list, tuple)):
        observed_rates = saved_observations
        print("Warning: legacy rate file has no cell-type metadata; using its stored ordering.")
    else:
        raise TypeError("--rates must contain rate tensors or rate tensors with cell-type metadata.")
    if len(observed_rates) != len(encoder.cell_type_names):
        raise ValueError("The number of observed-rate tensors does not match the selected cell types.")
    print(f"Loaded rates for: {', '.join(encoder.cell_type_names)}")
    print("Creating deterministic RGC likelihood model...")
    likelihood = DeterministicRateLikelihood(encoder, args.rate_noise_std)
    if args.backbone == "dps":
        print(f"Loading DPS prior: {args.dps_checkpoint} (preset {args.dps_preset})...")
    else:
        print(f"Loading unconditional DDPM prior: {args.model_id}...")
    reconstructor = DiffusionMAPReconstructor(
        likelihood, args.model_id, args.device,
        backbone=args.backbone,
        dps_checkpoint=args.dps_checkpoint,
        dps_preset=args.dps_preset,
    )
    print(
        f"Starting {args.steps}-step diffusion-MAP reconstruction on {args.device} "
        f"(guidance={args.guidance_scale}, normalization={args.guidance_normalization})..."
    )
    image, losses = reconstructor.reconstruct(
        observed_rates, args.steps, args.guidance_scale, args.guidance_normalization, args.seed
    )
    print(f"Saving reconstructed grayscale image to {args.output}...")
    image_uint8 = (image.squeeze(0).squeeze(0).cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    Image.fromarray(image_uint8.numpy(), mode="L").save(args.output)
    if args.loss_output:
        print(f"Saving likelihood trace to {args.loss_output}...")
        torch.save({"likelihood_loss": losses}, args.loss_output)
    print("Diffusion-MAP reconstruction complete.")


if __name__ == "__main__":
    main()
