"""Train one task-optimized RGC mosaic through per-step MAP guidance.

The image prior remains a frozen unconditional DDPM.  Under the default
``per_step_map`` mode every reverse step applies its MAP likelihood
correction, and at each such step the correction is differentiated with
respect to the learned mosaic and an optimizer update is taken.  The
supervision is image distance: whether that step's correction moved the
clean-image estimate toward the target.  The learned type is therefore
optimized for being *informative*, not for matching its own measurements.

Two single-update-per-trajectory modes are retained for reproducibility:
``final_stage`` guides only the last ``differentiable_steps`` updates, and
``random_step`` applies no guidance at all and scores rate mismatch at
randomly chosen timesteps.  Both train against rate mismatch, whose minimum
is an encoder invariant to its input; see NOTES.md.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from diffusion_retina import KERNEL_PARAMETERIZATIONS, LearnableSmoothMonostratifiedMosaic, StaticRGCEncoder
from sk_utils import Utils as u

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

class GrayscaleImageDataset(Dataset):
    """Load a file or directory of images as normalized grayscale tensors."""

    def __init__(self, target: str, image_size: int):
        """Collect supported image files and store the desired image resolution.

        Args:
            target: Path to a supported image file or a directory of images.
            image_size: Square output width and height in pixels.

        Returns:
            None.
        """
        # Resolve input paths once so DataLoader workers only decode images.
        root = Path(target)
        if root.is_file():
            self.paths = [root]
        elif root.is_dir():
            self.paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            raise FileNotFoundError(f"--target-image does not exist: {root}")
        if not self.paths:
            raise ValueError(f"No supported image files found in {root}.")
        self.image_size = image_size

    def __len__(self) -> int:
        """Return the number of available target images.

        Args:
            None.

        Returns:
            Number of images in this dataset.
        """
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Load one normalized grayscale image.

        Args:
            index: Dataset index of the requested image.

        Returns:
            Image tensor shaped ``(1, image_size, image_size)`` in ``[0, 1]``.
        """
        # Normalize image content before the batch dimension is added by DataLoader.
        with Image.open(self.paths[index]) as source:
            image = source.convert("L").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(pixels).unsqueeze(0)

def ddpm_rgb_to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB DDPM states from ``[-1, 1]`` to grayscale ``[0, 1]``.

    Args:
        rgb: RGB tensor shaped ``(B, 3, H, W)``.

    Returns:
        Grayscale tensor shaped ``(B, 1, H, W)``.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("The current unconditional DDPM prior requires RGB states shaped (B, 3, H, W).")
    weights = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return ((rgb * weights).sum(dim=1, keepdim=True) + 1.0).div(2.0).clamp(0, 1)

def predict_original_sample(scheduler: DDPMScheduler, sample: torch.Tensor, model_output: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """Predict the clean DDPM sample without breaking autograd.

    Args:
        scheduler: DDPM scheduler that defines the prediction convention.
        sample: Current RGB diffusion state shaped ``(B, 3, H, W)``.
        model_output: Frozen U-Net output shaped like ``sample``.
        timestep: Current scalar DDPM timestep.

    Returns:
        Predicted clean RGB sample shaped like ``sample``.
    """
    # Decode the U-Net target using the scheduler's configured prediction type.
    alpha_bar = scheduler.alphas_cumprod.to(sample.device)[timestep].to(sample.dtype)
    beta_bar = 1 - alpha_bar
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        predicted = (sample - beta_bar.sqrt() * model_output) / alpha_bar.sqrt()
    elif prediction_type == "sample":
        predicted = model_output
    elif prediction_type == "v_prediction":
        predicted = alpha_bar.sqrt() * sample - beta_bar.sqrt() * model_output
    else:
        raise ValueError(f"Unsupported DDPM prediction type: {prediction_type}")
    if scheduler.config.thresholding:
        raise NotImplementedError("Training sampler does not implement dynamic thresholding.")
    if scheduler.config.clip_sample:
        predicted = predicted.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)
    return predicted

def predicted_epsilon(scheduler: DDPMScheduler, sample: torch.Tensor, model_output: torch.Tensor, predicted_x0: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """Convert a scheduler-specific U-Net output to epsilon prediction.

    Args:
        scheduler: DDPM scheduler that defines the prediction convention.
        sample: Current RGB diffusion state shaped ``(B, 3, H, W)``.
        model_output: Frozen U-Net output shaped like ``sample``.
        predicted_x0: Predicted clean RGB sample shaped like ``sample``.
        timestep: Current scalar DDPM timestep.

    Returns:
        Epsilon tensor shaped like ``sample``.
    """
    # Recover epsilon so a single deterministic DDIM update handles all modes.
    alpha_bar = scheduler.alphas_cumprod.to(sample.device)[timestep].to(sample.dtype)
    beta_bar = 1 - alpha_bar
    kind = scheduler.config.prediction_type
    if kind == "epsilon":
        return model_output
    if kind == "sample":
        return (sample - alpha_bar.sqrt() * predicted_x0) / beta_bar.sqrt().clamp_min(1e-8)
    if kind == "v_prediction":
        return alpha_bar.sqrt() * model_output + beta_bar.sqrt() * sample
    raise ValueError(f"Unsupported DDPM prediction type: {kind}")

def ddim_step(scheduler: DDPMScheduler, sample: torch.Tensor, model_output: torch.Tensor, timestep: torch.Tensor, previous_timestep: torch.Tensor | int) -> torch.Tensor:
    """Apply one deterministic, differentiable DDIM reverse update.

    Args:
        scheduler: DDPM scheduler containing cumulative alphas.
        sample: Current RGB diffusion state shaped ``(B, 3, H, W)``.
        model_output: Frozen U-Net output shaped like ``sample``.
        timestep: Current scalar DDPM timestep.
        previous_timestep: Next timestep in the reverse schedule or ``-1``.

    Returns:
        Previous RGB diffusion state shaped like ``sample``.
    """
    # Build the eta=0 update explicitly to preserve the autograd graph.
    predicted_x0 = predict_original_sample(scheduler, sample, model_output, timestep)
    epsilon = predicted_epsilon(scheduler, sample, model_output, predicted_x0, timestep)
    alphas = scheduler.alphas_cumprod.to(sample.device, dtype=sample.dtype)
    alpha_prev = sample.new_tensor(1.0) if int(previous_timestep) < 0 else alphas[previous_timestep]
    return alpha_prev.sqrt() * predicted_x0 + (1 - alpha_prev).sqrt() * epsilon

class FrozenAndLearnedRateLikelihood(torch.nn.Module):
    """RGC rate loss whose only trainable path is the appended learned mosaic."""

    def __init__(self, fixed_encoder: StaticRGCEncoder, learned_encoder: LearnableSmoothMonostratifiedMosaic, rate_noise_std: float, learned_weight: float = 1.0):
        """Initialize frozen and learned components of the rate likelihood.

        Args:
            fixed_encoder: Frozen collection of established RGC mosaics.
            learned_encoder: Task-optimized extra RGC mosaic.
            rate_noise_std: Gaussian surrogate standard deviation for rates.
            learned_weight: Relative weight of the learned mosaic's rate term.

        Returns:
            None.
        """
        super().__init__()
        if rate_noise_std <= 0:
            raise ValueError("rate_noise_std must be positive.")
        if learned_weight <= 0:
            raise ValueError("learned_weight must be positive.")
        self.fixed_encoder = fixed_encoder
        self.learned_encoder = learned_encoder
        self.rate_noise_std = float(rate_noise_std)
        # A plain mean over mosaics gives the new cell type only 1/n of the
        # likelihood; this restores control over its share.
        self.learned_weight = float(learned_weight)
        self._fixed_rates: list[torch.Tensor] | None = None

    def cache_fixed_rates(self, target: torch.Tensor) -> None:
        """Encode a target batch once with the frozen mosaics.

        Args:
            target: Target grayscale images shaped ``(B, 1, H, W)``.

        Returns:
            None.
        """
        # The frozen mosaics do not change within a batch, so their target
        # measurements are computed once and reused at every MAP step.
        with torch.no_grad():
            self._fixed_rates = [rates.detach() for rates in self.fixed_encoder(target)]

    def observed_rates(self, target: torch.Tensor, attach_learned: bool = True) -> list[torch.Tensor]:
        """Generate target-image rate observations with selective gradients.

        Args:
            target: Target grayscale images shaped ``(B, 1, H, W)``.
            attach_learned: Retain the learned mosaic's graph for training steps.

        Returns:
            Rate tensors for fixed mosaics followed by the learned rate.
        """
        if self._fixed_rates is None:
            self.cache_fixed_rates(target)
        if attach_learned:
            # Recomputed fresh at every training step: an optimizer update
            # changes theta, so a cached measurement is both stale and backed
            # by a graph that the previous backward pass already freed.
            learned = self.learned_encoder(target)
        else:
            with torch.no_grad():
                learned = self.learned_encoder(target).detach()
        return [*self._fixed_rates, learned]

    def loss(self, image: torch.Tensor, observed_rates: list[torch.Tensor]) -> torch.Tensor:
        """Compute batched Gaussian-surrogate RGC rate mismatch.

        Args:
            image: Predicted grayscale images shaped ``(B, 1, H, W)``.
            observed_rates: Target rate tensor for every fixed and learned mosaic.

        Returns:
            Scalar rate-matching loss.
        """
        # Re-encode the candidate image through every measurement channel.
        predicted = [*self.fixed_encoder(image), self.learned_encoder(image)]
        if len(predicted) != len(observed_rates):
            raise ValueError("Predicted and observed RGC mosaics do not match.")
        weights = [1.0] * (len(predicted) - 1) + [self.learned_weight]
        terms = [
            weight * (value - observed.to(value)).square().mean()
            for weight, value, observed in zip(weights, predicted, observed_rates)
        ]
        return torch.stack(terms).sum() / sum(weights) / (2 * self.rate_noise_std**2)

def guidance_correction(gradient: torch.Tensor, scale: float, normalization: str) -> torch.Tensor:
    """Scale a likelihood gradient before applying MAP guidance.

    Args:
        gradient: Image-space likelihood gradient.
        scale: Requested guidance magnitude.
        normalization: Either raw scaling or RMS normalization.

    Returns:
        Scaled guidance correction tensor.
    """
    if normalization == "none":
        return scale * gradient
    if normalization == "rms":
        return scale * gradient / gradient.square().mean().sqrt().clamp_min(1e-12)
    raise ValueError(f"Unsupported guidance normalization: {normalization}")


def differentiable_unet_prediction(unet: UNet2DModel, sample: torch.Tensor, timestep: torch.Tensor, mixed_precision: bool, force_math_sdp: bool) -> torch.Tensor:
    """Run a differentiable U-Net pass with an optional double-backward-safe SDP kernel.

    Args:
        unet: Frozen unconditional RGB denoising U-Net.
        sample: Current RGB diffusion state shaped ``(B, 3, H, W)``.
        timestep: Current scalar DDPM timestep.
        mixed_precision: Whether to autocast CUDA U-Net computation.
        force_math_sdp: Disable Flash and memory-efficient SDP attention kernels.

    Returns:
        U-Net prediction tensor shaped like ``sample``.
    """
    # Math SDP supports the double backward required by exact guidance training.
    amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if mixed_precision else nullcontext()
    if force_math_sdp and sample.device.type == "cuda":
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            with amp_context:
                return unet(sample, timestep).sample
    with amp_context:
        return unet(sample, timestep).sample


def reconstruct_with_final_stage_gradients(unet: UNet2DModel, scheduler: DDPMScheduler, likelihood: FrozenAndLearnedRateLikelihood, observed_rates: list[torch.Tensor], batch_size: int, guidance_scale: float, guidance_normalization: str, differentiable_steps: int, generator: torch.Generator, mixed_precision: bool, force_math_sdp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct an image batch while retaining only final MAP-step gradients.

    Args:
        unet: Frozen unconditional RGB denoising U-Net.
        scheduler: DDPM schedule used to create deterministic DDIM updates.
        likelihood: Fixed-plus-learned RGC rate likelihood.
        observed_rates: Target-image rate observations for every mosaic.
        batch_size: Number of independently reconstructed images.
        guidance_scale: MAP guidance magnitude.
        guidance_normalization: Either ``none`` or ``rms``.
        differentiable_steps: Number of final reverse updates retaining a graph.
        generator: Seeded random generator for initial diffusion noise.
        mixed_precision: Whether to autocast CUDA U-Net computation.
        force_math_sdp: Use math SDP attention for final double-backward steps.

    Returns:
        Final grayscale reconstruction batch and the final RGC likelihood loss.
    """
    if differentiable_steps < 1 or differentiable_steps > len(scheduler.timesteps):
        raise ValueError("--differentiable-steps must be in [1, num_inference_steps].")
    # Initialize the unconditional RGB trajectory and find the graph boundary.
    sample_size = unet.config.sample_size
    device = next(unet.parameters()).device
    sample = torch.randn((batch_size, 3, sample_size, sample_size), device=device, generator=generator)
    timesteps = scheduler.timesteps
    first_differentiable = len(timesteps) - differentiable_steps
    for index, timestep in enumerate(timesteps):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        if index < first_differentiable:
            # Produce an unconditional prefix without retaining its activations.
            with torch.no_grad():
                prediction = differentiable_unet_prediction(unet, sample, timestep, mixed_precision, False)
                sample = ddim_step(scheduler, sample, prediction, timestep, previous)
            continue

        # Unroll only the final MAP updates and preserve higher-order gradients.
        sample = sample.requires_grad_(True)
        prediction = differentiable_unet_prediction(unet, sample, timestep, mixed_precision, force_math_sdp)
        predicted_x0 = predict_original_sample(scheduler, sample, prediction, timestep)
        rate_loss = likelihood.loss(ddpm_rgb_to_grayscale(predicted_x0), observed_rates)
        (rate_gradient,) = torch.autograd.grad(rate_loss, sample, create_graph=True)
        correction = guidance_correction(rate_gradient, guidance_scale, guidance_normalization)
        sample = ddim_step(scheduler, sample, prediction, timestep, previous) - correction

    return ddpm_rgb_to_grayscale(sample), rate_loss


def reconstruct_with_random_step_rate_loss(
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
    likelihood: FrozenAndLearnedRateLikelihood,
    observed_rates: list[torch.Tensor],
    batch_size: int,
    n_gradient_steps: int,
    generator: torch.Generator,
    mixed_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a full no-grad DDIM trajectory; compute rate loss at n random steps.

    The full reverse diffusion trajectory is run without retaining any
    computation graph.  At each of ``n_gradient_steps`` randomly selected
    timesteps the denoiser state is stored; rate mismatch losses are then
    computed independently for each stored state so that gradient flows only
    through the learned encoder parameters, not through the diffusion trajectory.

    This is more memory-efficient than ``reconstruct_with_final_stage_gradients``
    because only one UNet pass worth of activations is live at a time.

    Args:
        unet: Frozen unconditional RGB denoising U-Net.
        scheduler: DDPM schedule used to create deterministic DDIM updates.
        likelihood: Fixed-plus-learned RGC rate likelihood.
        observed_rates: Target-image rate observations for every mosaic.
        batch_size: Number of independently reconstructed images.
        n_gradient_steps: Number of randomly sampled timesteps for rate loss.
        generator: Seeded random generator for initial diffusion noise.
        mixed_precision: Whether to autocast CUDA U-Net computation.

    Returns:
        Final grayscale reconstruction batch (no gradient) and mean rate loss
        averaged across the randomly selected timesteps.
    """
    sample_size = unet.config.sample_size
    device = next(unet.parameters()).device
    sample = torch.randn((batch_size, 3, sample_size, sample_size), device=device, generator=generator)
    timesteps = scheduler.timesteps
    n = min(int(n_gradient_steps), len(timesteps))
    selected_indices = set(random.sample(range(len(timesteps)), n))

    # Run the full reverse trajectory without retaining any computation graph.
    # Store a detached copy of the state at each randomly selected timestep.
    stored: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for index, timestep in enumerate(timesteps):
            previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            prediction = differentiable_unet_prediction(unet, sample, timestep, mixed_precision, False)
            if index in selected_indices:
                stored.append((sample.clone(), timestep))
            sample = ddim_step(scheduler, sample, prediction, timestep, previous)

    final_reconstruction = ddpm_rgb_to_grayscale(sample)

    # For each stored state compute the rate mismatch independently.
    # predicted_x0 is derived from a detached state and a frozen UNet, so it
    # carries no gradient w.r.t. the diffusion trajectory.  Gradient flows
    # solely through the learned encoder weights inside likelihood.loss().
    rate_losses: list[torch.Tensor] = []
    for x_t, timestep in stored:
        prediction = differentiable_unet_prediction(unet, x_t, timestep, mixed_precision, False)
        predicted_x0 = predict_original_sample(scheduler, x_t, prediction, timestep)
        rate_loss = likelihood.loss(ddpm_rgb_to_grayscale(predicted_x0), observed_rates)
        rate_losses.append(rate_loss)

    return final_reconstruction, torch.stack(rate_losses).mean()


def step_loss_weight(alpha_prev: torch.Tensor, weighting: str) -> torch.Tensor:
    """Weight one MAP step's image loss by its position on the noise schedule.

    Args:
        alpha_prev: Cumulative alpha at the step being written to.
        weighting: One of ``uniform``, ``alpha_prev``, or ``snr``.

    Returns:
        Scalar weight tensor.
    """
    # Early, noisy steps carry a large 1/sqrt(alpha_prev) factor in the
    # corrected clean estimate; downweighting them keeps the image loss
    # commensurate with the regularizers across the trajectory.
    if weighting == "uniform":
        return torch.ones_like(alpha_prev)
    if weighting == "alpha_prev":
        return alpha_prev
    if weighting == "snr":
        return (alpha_prev / (1 - alpha_prev).clamp_min(1e-8)).clamp(max=1e3)
    raise ValueError(f"Unsupported step loss weighting: {weighting}")


def reconstruct_with_per_step_map_updates(unet: UNet2DModel, scheduler: DDPMScheduler, likelihood: FrozenAndLearnedRateLikelihood, target: torch.Tensor, guidance_scale: float, guidance_normalization: str, optimizer: torch.optim.Optimizer, grad_clip_norm: float, update_every: int, weighting: str, regularization: dict[str, float], generator: torch.Generator, mixed_precision: bool, force_math_sdp: bool) -> dict[str, float]:
    """Run a fully MAP-guided trajectory, updating the learned mosaic each step.

    Guidance is applied at every reverse step, so the trajectory is genuinely
    conditioned on the observed rates.  At every ``update_every``-th step the
    correction is additionally differentiated with respect to the learned
    encoder and an optimizer update is taken, so the new cell type is trained
    on whether its measurement gradient actually moves the clean-image
    estimate toward the target.

    Args:
        unet: Frozen unconditional RGB denoising U-Net.
        scheduler: DDPM schedule supplying cumulative alphas.
        likelihood: Fixed-plus-learned RGC rate likelihood.
        target: Target grayscale images shaped ``(B, 1, H, W)``.
        guidance_scale: MAP guidance magnitude applied at every step.
        guidance_normalization: Either ``none`` or ``rms``.
        optimizer: Optimizer over the learned mosaic's parameters.
        grad_clip_norm: Gradient-norm clip applied before each update.
        update_every: Take a training update every n-th reverse step.
        weighting: Schedule weighting applied to each step's image loss.
        regularization: Weights for ``smoothness``, ``diversity``, ``rate_budget``.
        generator: Seeded generator for the initial diffusion noise.
        mixed_precision: Whether to autocast CUDA U-Net computation.
        force_math_sdp: Use math SDP attention on double-backward steps.

    Returns:
        Metrics for this trajectory, including per-step and final image losses.
    """
    if update_every < 1:
        raise ValueError("--map-update-every must be positive.")
    device = next(unet.parameters()).device
    batch_size = target.shape[0]
    sample_size = unet.config.sample_size
    sample = torch.randn((batch_size, 3, sample_size, sample_size), device=device, generator=generator)
    timesteps = scheduler.timesteps
    alphas = scheduler.alphas_cumprod.to(device)

    image_losses: list[float] = []
    rate_losses: list[float] = []

    for index, timestep in enumerate(timesteps):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        alpha_prev = torch.ones((), device=device) if int(previous) < 0 else alphas[previous]
        trains = index % update_every == 0

        # Every step needs d(rate loss)/d(x_t) to form its correction.  Only
        # training steps additionally retain the second-order path that
        # reaches the learned encoder's parameters through that gradient.
        sample = sample.detach().requires_grad_(True)
        prediction = differentiable_unet_prediction(unet, sample, timestep, mixed_precision, force_math_sdp and trains)
        predicted_x0 = predict_original_sample(scheduler, sample, prediction, timestep)
        observed = likelihood.observed_rates(target, attach_learned=trains)
        rate_loss = likelihood.loss(ddpm_rgb_to_grayscale(predicted_x0), observed)
        (rate_gradient,) = torch.autograd.grad(rate_loss, sample, create_graph=trains)
        correction = guidance_correction(rate_gradient, guidance_scale, guidance_normalization)
        next_sample = ddim_step(scheduler, sample, prediction, timestep, previous) - correction

        if trains:
            # Map the corrected state back to clean-image space by reusing this
            # step's epsilon: x0_corrected = x0 - correction / sqrt(alpha_prev).
            # The loss therefore asks directly whether this correction moved
            # the clean estimate toward the target, with no second U-Net pass.
            # ``predicted_x0`` is detached because theta reaches this loss only
            # through ``correction``.
            corrected_x0 = predicted_x0.detach() - correction / alpha_prev.sqrt().clamp_min(1e-8)
            image_loss = (ddpm_rgb_to_grayscale(corrected_x0) - target).square().mean()
            smoothness, diversity = likelihood.learned_encoder.regularization()
            mean_rate = observed[-1].mean()
            total = (
                step_loss_weight(alpha_prev, weighting) * image_loss
                + regularization["smoothness"] * smoothness
                + regularization["diversity"] * diversity
                + regularization["rate_budget"] * mean_rate.square()
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(likelihood.learned_encoder.parameters(), grad_clip_norm)
            optimizer.step()
            image_losses.append(float(image_loss.detach()))

        rate_losses.append(float(rate_loss.detach()))
        sample = next_sample.detach()

    reconstruction = ddpm_rgb_to_grayscale(sample)
    final_mse = float((reconstruction - target).square().mean())
    return {
        "reconstruction": reconstruction,
        "step_image_loss": sum(image_losses) / max(len(image_losses), 1),
        "rate_loss": sum(rate_losses) / max(len(rate_losses), 1),
        "final_mse": final_mse,
        "n_updates": len(image_losses),
    }


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators.

    Args:
        seed: Integer seed shared by all local random generators.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_preview(image: torch.Tensor, path: Path) -> None:
    """Save the first grayscale image in a reconstruction batch as PNG.

    Args:
        image: Grayscale reconstruction batch shaped ``(B, 1, H, W)``.
        path: Destination PNG path.

    Returns:
        None.
    """
    pixels = (image[0, 0].detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    Image.fromarray(pixels, mode="L").save(path)


def save_learned_weights(path: Path, epoch: int, learned_encoder: LearnableSmoothMonostratifiedMosaic) -> None:
    """Append an interpretable learned-mosaic weight snapshot to an HDF5 file.

    Args:
        path: Destination HDF5 file shared by all saved epochs.
        epoch: Completed training epoch represented by these parameters.
        learned_encoder: Post-update learned smooth-monostratified mosaic.

    Returns:
        None.
    """
    # Store every save point under an independently inspectable epoch group.
    with h5py.File(path, "a") as handle:
        handle.attrs["format"] = "diffusion_retina_learned_mosaic_weights_v1"
        group_name = f"epoch_{epoch:04d}"
        if group_name in handle:
            del handle[group_name]
        group = handle.create_group(group_name)
        group.attrs["epoch"] = epoch

        # Save both optimized functional kernels and their unconstrained parameters.
        kernels = learned_encoder.kernels.detach().cpu().float()
        group.create_dataset("functional_subunit_kernels", data=kernels.numpy(), compression="gzip")
        group.create_dataset("raw_subunit_kernel_parameters", data=learned_encoder.raw_kernels.detach().cpu().float().numpy(), compression="gzip")
        group.create_dataset("summed_subunit_kernel", data=kernels.sum(dim=0).numpy(), compression="gzip")
        group.create_dataset("cell_centres", data=learned_encoder.centres.detach().cpu().float().numpy(), compression="gzip")

        # Record scalar response parameters needed to interpret each kernel snapshot.
        for name in ("log_subunit_alpha", "log_subunit_beta", "subunit_gamma", "log_output_alpha", "log_output_beta", "output_gamma"):
            group.create_dataset(name, data=getattr(learned_encoder, name).detach().cpu().float().numpy())

def parse_args() -> argparse.Namespace:
    """Parse CLI options and merge general and learning-specific YAML defaults.

    Args:
        None.

    Returns:
        Namespace where CLI values override learning and base diffusion YAML.
    """
    # Read config locations first, then apply their values to unset CLI fields.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="diffusion_params_train.yaml")
    bootstrap.add_argument("--diffusion-params", default=None)
    config_path, _ = bootstrap.parse_known_args()
    learn_config = u.read_params(config_path.config) if Path(config_path.config).is_file() else {}
    base_path = config_path.diffusion_params or learn_config.get("diffusion_params", "diffusion_params.yaml")
    base_config = u.read_params(base_path) if Path(base_path).is_file() else {}
    # General diffusion defaults < learning YAML < explicit CLI values.
    config = {**base_config, **learn_config}

    # Declare every run-time override accepted by the training entry point.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=config_path.config)
    parser.add_argument("--target-image", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--diffusion-params", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--differentiable-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--guidance-normalization", choices=("none", "rms"), default=None)
    parser.add_argument("--rate-noise-std", type=float, default=None)
    parser.add_argument("--learn-cell-type", default=None)
    parser.add_argument("--learned-n-cells", type=int, default=None)
    parser.add_argument("--learned-init-seed", type=int, default=None)
    parser.add_argument("--include-cell-types", nargs="+", default=None)
    parser.add_argument("--rf-smoothness-weight", type=float, default=None)
    parser.add_argument("--rate-budget-weight", type=float, default=None)
    parser.add_argument("--diversity-weight", type=float, default=None)
    parser.add_argument("--output-every-epochs", type=int, default=None)
    parser.add_argument("--weight-save-every-epochs", type=int, default=None)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force-math-sdp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--n-gradient-steps", type=int, default=None)
    parser.add_argument("--train-mode", choices=("per_step_map", "random_step", "final_stage"), default=None,
                        help="per_step_map guides every reverse step and updates the learned mosaic at each one.")
    parser.add_argument("--map-update-every", type=int, default=None,
                        help="Take a learned-mosaic update every n-th guided reverse step.")
    parser.add_argument("--step-loss-weighting", choices=("uniform", "alpha_prev", "snr"), default=None)
    parser.add_argument("--learned-weight", type=float, default=None,
                        help="Relative weight of the learned mosaic in the rate likelihood.")
    parser.add_argument("--kernel-parameterization", choices=KERNEL_PARAMETERIZATIONS, default=None)
    parser.add_argument("--learned-subunit-nonlinearity", default=None)
    parser.add_argument("--learned-output-nonlinearity", default=None)
    parser.add_argument("--learned-rf-scale", type=float, default=None,
                        help="Scale the learned cell type's spatial lengths only (0.25 for 64px).")
    args = parser.parse_args()
    for key, value in config.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)
    if args.force_math_sdp is None:
        args.force_math_sdp = True
    defaults = {
        "train_mode": "per_step_map",
        "map_update_every": 1,
        "step_loss_weighting": "alpha_prev",
        "learned_weight": 1.0,
        "kernel_parameterization": "zero_mean_unit_norm",
        "learned_rf_scale": 1.0,
    }
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    required = ("target_image", "output_dir", "params", "model_id", "image_size", "device")
    missing = [name for name in required if getattr(args, name, None) is None]
    if missing:
        parser.error(f"Missing required settings (CLI or {args.config}): {', '.join(missing)}")
    return args

def main() -> None:
    """Run task-optimized learned-mosaic training from configured image data.

    Args:
        None.

    Returns:
        None.
    """
    # Resolve configuration, deterministic state, and output persistence.
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(args.device)
    if args.mixed_precision and device.type != "cuda":
        raise ValueError("--mixed-precision requires a CUDA device.")
    print(f"Final differentiable attention backend: {'math SDP' if args.force_math_sdp else 'default SDP'}.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    # Build the target-image batches and frozen/learned retinal measurements.
    dataset = GrayscaleImageDataset(args.target_image, int(args.image_size))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), pin_memory=device.type == "cuda")
    retina_params = u.read_params(args.params)
    fixed_types = list(args.include_cell_types or [])
    if args.learn_cell_type in fixed_types:
        raise ValueError("--learn-cell-type must not also appear in --include-cell-types.")
    fixed_encoder = StaticRGCEncoder.from_params(retina_params, (args.image_size, args.image_size), fixed_types).to(device).eval()
    learned_encoder = LearnableSmoothMonostratifiedMosaic.from_params(
        retina_params, (args.image_size, args.image_size), args.learn_cell_type, args.learned_init_seed, args.learned_n_cells,
        kernel_parameterization=args.kernel_parameterization,
        subunit_nonlinearity=args.learned_subunit_nonlinearity,
        output_nonlinearity=args.learned_output_nonlinearity,
        rf_scale=float(args.learned_rf_scale),
    ).to(device).train()
    print(
        f"Learned mosaic: {learned_encoder.centres.shape[0]} cells, "
        f"{tuple(int(v) for v in learned_encoder.rf_shape)} px RF, "
        f"{learned_encoder.n_subunits} subunits, "
        f"{args.kernel_parameterization} kernels, "
        f"{learned_encoder.subunit_kind}/{learned_encoder.output_kind} nonlinearities."
    )
    if learned_encoder.centres.shape[0] < 16:
        print(
            f"Warning: only {learned_encoder.centres.shape[0]} learned cells at {args.image_size}px. "
            "Rescale the learned cell type's spatial parameters for this image size."
        )
    likelihood = FrozenAndLearnedRateLikelihood(fixed_encoder, learned_encoder, args.rate_noise_std, args.learned_weight).to(device)

    # Load and freeze the existing unconditional image prior.
    unet = UNet2DModel.from_pretrained(args.model_id).to(device).eval()
    if int(unet.config.sample_size) != int(args.image_size):
        raise ValueError(
            f"--image-size ({args.image_size}) must match the unconditional checkpoint sample size "
            f"({unet.config.sample_size})."
        )
    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(args.model_id)
    scheduler.set_timesteps(int(args.num_inference_steps), device=device)
    optimizer = torch.optim.AdamW(learned_encoder.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    best_loss = float("inf")
    history: list[dict[str, float]] = []

    # Track every batch in one persistent progress bar across the full run.
    total_epochs = int(args.epochs)
    output_every_epochs = int(args.output_every_epochs or 20)
    weight_save_every_epochs = int(args.weight_save_every_epochs or 20)
    if output_every_epochs < 1:
        raise ValueError("--output-every-epochs must be positive.")
    if weight_save_every_epochs < 1:
        raise ValueError("--weight-save-every-epochs must be positive.")
    progress = tqdm(total=total_epochs * len(loader), desc="Training", unit="batch")
    regularization = {
        "smoothness": float(args.rf_smoothness_weight),
        "diversity": float(args.diversity_weight),
        "rate_budget": float(args.rate_budget_weight),
    }
    print(f"Training mode: {args.train_mode} (update every {args.map_update_every} guided step(s)).")

    # Optimize only the extra RGC mosaic against guided-reconstruction quality.
    for epoch in range(1, total_epochs + 1):
        epoch_mse = epoch_rate = epoch_total = 0.0
        save_weights = epoch % weight_save_every_epochs == 0 or epoch == total_epochs
        progress.set_postfix(epoch=f"{epoch}/{total_epochs}")
        for target in loader:
            target = target.to(device, non_blocking=True)
            likelihood.cache_fixed_rates(target)

            if args.train_mode == "per_step_map":
                # Guidance at every reverse step, one update per MAP step.
                stats = reconstruct_with_per_step_map_updates(
                    unet, scheduler, likelihood, target, args.guidance_scale,
                    args.guidance_normalization, optimizer, args.grad_clip_norm,
                    int(args.map_update_every), args.step_loss_weighting, regularization,
                    generator, bool(args.mixed_precision), bool(args.force_math_sdp),
                )
                reconstruction = stats["reconstruction"]
                epoch_mse += stats["final_mse"]
                epoch_rate += stats["rate_loss"]
                epoch_total += stats["step_image_loss"]
            else:
                # Legacy single-update-per-trajectory paths, retained so the
                # earlier runs stay reproducible.
                observed = likelihood.observed_rates(target)
                if args.train_mode == "random_step":
                    reconstruction, rate_loss = reconstruct_with_random_step_rate_loss(
                        unet, scheduler, likelihood, observed, target.shape[0],
                        args.n_gradient_steps, generator, bool(args.mixed_precision),
                    )
                else:
                    reconstruction, rate_loss = reconstruct_with_final_stage_gradients(
                        unet, scheduler, likelihood, observed, target.shape[0],
                        args.guidance_scale, args.guidance_normalization,
                        int(args.differentiable_steps), generator,
                        bool(args.mixed_precision), bool(args.force_math_sdp),
                    )
                with torch.no_grad():
                    mse = (reconstruction - target).square().mean()
                smoothness, diversity = learned_encoder.regularization()
                mean_rate = observed[-1].mean()
                total = (rate_loss + regularization["smoothness"] * smoothness
                         + regularization["diversity"] * diversity
                         + regularization["rate_budget"] * mean_rate.square())
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                if args.grad_clip_norm and args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(learned_encoder.parameters(), args.grad_clip_norm)
                optimizer.step()
                epoch_mse += float(mse.detach())
                epoch_rate += float(rate_loss.detach())
                epoch_total += float(total.detach())
            progress.update(1)

        # Calculate metrics every epoch, but write artifacts only at the requested interval.
        n_batches = len(loader)
        metrics = {"epoch": epoch, "mse": epoch_mse / n_batches, "rgc_loss": epoch_rate / n_batches, "total_loss": epoch_total / n_batches}
        history.append(metrics)
        should_output = epoch % output_every_epochs == 0 or epoch == total_epochs
        if should_output:
            progress.set_postfix(epoch=f"{epoch}/{total_epochs}", mse=f"{metrics['mse']:.5g}", rgc_loss=f"{metrics['rgc_loss']:.5g}")
            print(json.dumps(metrics))
            state = {"epoch": epoch, "args": vars(args), "learned_encoder_state_dict": learned_encoder.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "history": history}
            torch.save(state, output_dir / "last.pt")
            if metrics["total_loss"] < best_loss:
                best_loss = metrics["total_loss"]
                torch.save(state, output_dir / "best.pt")
            save_preview(target, output_dir / f"target_epoch_{epoch:04d}.png")
            save_preview(reconstruction, output_dir / f"reconstruction_epoch_{epoch:04d}.png")
            torch.save(state, output_dir / f"checkpoint_epoch_{epoch:04d}.pt")
        if save_weights:
            save_learned_weights(output_dir / "learned_mosaic_weights.h5", epoch, learned_encoder)
    progress.close()

if __name__ == "__main__":
    main()
