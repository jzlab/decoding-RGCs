"""Search RMS-normalized RGC-guidance strength across reverse-diffusion trajectories."""

from __future__ import annotations

# Running this file directly puts its own folder on sys.path, not the repository
# root, so the shared package would not resolve. Prepend the root before any
# first-party import. Harmless when the module is imported as part of a package.
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = str(_Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import os
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import lpips

from diffusion.diffusion_map import DeterministicRateLikelihood
from diffusion.diffusion_reconstruct import DiffusionMAPReconstructor, ddpm_rgb_to_grayscale
from diffusion.diffusion_retina import StaticRGCEncoder
from shared.sk_utils import Utils as u


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def collect_target_images(target_path: Path) -> tuple[list[Path], bool]:
    """Resolve an image input or collect direct-child images from an input folder.

    Args:
        target_path (Path): Existing image file or directory containing image files.

    Returns:
        tuple[list[Path], bool]: Sorted image paths and whether the input was a directory.
    """
    if target_path.is_file():
        if target_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported target image type: {target_path.suffix}")
        return [target_path], False
    if target_path.is_dir():
        image_paths = sorted(
            path for path in target_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise ValueError(f"No supported image files found in {target_path}.")
        return image_paths, True
    raise FileNotFoundError(f"--target-image does not exist: {target_path}")


def source_name(image_path: Path, target_root: Path, folder_mode: bool) -> str:
    """Produce a stable source-image identifier for output CSV rows.

    Args:
        image_path (Path): Source image currently being searched.
        target_root (Path): Original path passed through ``--target-image``.
        folder_mode (bool): Whether the target root is a directory.

    Returns:
        str: Relative image path in folder mode or image path in single-image mode.
    """
    return str(image_path.relative_to(target_root)) if folder_mode else str(image_path)


def load_grayscale_image(image_path: str, image_size: int) -> torch.Tensor:
    """Load, grayscale-convert, and resize a target image to the search resolution.

    Args:
        image_path (str): Path to a PNG, JPEG, or other Pillow-readable image.
        image_size (int): Required square image resolution.

    Returns:
        torch.Tensor: Target image in [0, 1] with shape (1, 1, image_size, image_size).
    """
    with Image.open(image_path) as source:
        image = source.convert("L").resize((image_size, image_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)


def eta_grid() -> list[float]:
    """Create the requested zero-plus-logarithmic RGC-guidance grid.

    Args:
        None: This function accepts no arguments.

    Returns:
        list[float]: Zero followed by nine log-spaced values from 1e-4 to 0.25.
    """
    return [0.0, *np.geomspace(1e-4, 0.25, num=9).tolist()]


def timestep_grid() -> list[int]:
    """Create the requested reverse-update checkpoint grid.

    Args:
        None: This function accepts no arguments.

    Returns:
        list[int]: Completed-update counts from 5 through 100 in increments of 5.
    """
    return list(range(5, 101, 5))


def parameter_tag(eta: float, sigma_r: float, completed_steps: int) -> str:
    """Format hyperparameters into a stable image filename suffix.

    Args:
        eta (float): RGC-guidance strength.
        sigma_r (float): Assumed RGC-rate uncertainty.
        completed_steps (int): Number of completed reverse updates.

    Returns:
        str: Filesystem-safe hyperparameter tag.
    """
    eta_tag = "0" if eta == 0 else f"{eta:.4g}".replace(".", "p").replace("-", "m")
    sigma_tag = f"{sigma_r:g}".replace(".", "p")
    return f"eta_{eta_tag}__sigma_r_{sigma_tag}__T_{completed_steps:03d}"


def save_grayscale_image(image: torch.Tensor, output_path: Path) -> None:
    """Save a single grayscale tensor as an 8-bit PNG image.

    Args:
        image (torch.Tensor): Image in [0, 1] shaped (1, 1, height, width).
        output_path (Path): Destination PNG file path.

    Returns:
        None: The image is written to disk.
    """
    image_uint8 = (image.squeeze(0).squeeze(0).cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    Image.fromarray(image_uint8.numpy(), mode="L").save(output_path)


def run_trajectory(reconstructor: DiffusionMAPReconstructor, observed_rates: list[torch.Tensor], target_image: torch.Tensor, lpips_metric: lpips.LPIPS, eta: float, completed_steps_to_save: set[int], seed: int):
    """Run one RMS-normalized guidance trajectory and yield noisy scheduler outputs.

    Args:
        reconstructor (DiffusionMAPReconstructor): Loaded DDPM and RGC likelihood models.
        observed_rates (list[torch.Tensor]): RGC rates generated from the target image.
        target_image (torch.Tensor): Resized grayscale target image in [0, 1].
        eta (float): RMS-normalized RGC-guidance strength.
        completed_steps_to_save (set[int]): Reverse-update counts to yield.
        seed (int): Random seed used for DDPM initial and scheduler noise.

    Yields:
        tuple[int, torch.Tensor, float, float]: Completed steps, direct x_{t-1} image,
            RGC loss, and MSE at each requested checkpoint.
    """
    
    # get models ready for reconstructions
    device = reconstructor.device
    generator = torch.Generator(device=device).manual_seed(seed)
    sample_size = reconstructor.unet.config.sample_size
    sample = torch.randn((1, 3, sample_size, sample_size), device=device, generator=generator)
    scheduler = reconstructor.scheduler
    scheduler.set_timesteps(100, device=device)

    for completed_steps, timestep in enumerate(scheduler.timesteps, start=1):
        sample = sample.detach().requires_grad_(True)
        noise_prediction = reconstructor.unet(sample, timestep).sample
        predicted_clean_rgb = reconstructor._predict_original_sample(sample, noise_prediction, timestep)
        likelihood_loss = reconstructor.likelihood.loss( ddpm_rgb_to_grayscale(predicted_clean_rgb), observed_rates )
        (guidance_gradient,) = torch.autograd.grad(likelihood_loss, sample)

        with torch.no_grad():
            # use the ddpm to get the next unbiased step
            prior = scheduler.step( noise_prediction.detach(), timestep, sample.detach(), generator=generator )
            
            # RMS normalization gives every likelihood correction an RMS of eta. The saved state remains direct x_{t-1}, never predicted x_0.
            normalized_gradient = guidance_gradient / guidance_gradient.square().mean().sqrt().clamp_min(1e-12)
            sample = prior.prev_sample - eta * normalized_gradient

            if completed_steps not in completed_steps_to_save:
                continue

            # This is deliberately the direct guided noisy state x_{t-1}, not the U-Net's predicted x_0 used above to define the likelihood.
            noisy_output = ddpm_rgb_to_grayscale(sample)
            
            # compute our loss/tracking metrics
            final_rgc_loss = reconstructor.likelihood.loss(noisy_output, observed_rates)
            mse = (noisy_output - target_image).square().mean()
            lpips_score = lpips_metric(target_image.repeat(1, 3, 1, 1) * 2 - 1, noisy_output.repeat(1, 3, 1, 1) * 2 - 1)
            
            yield completed_steps, noisy_output.cpu(), float(final_rgc_loss.cpu()), float(mse.cpu()), float(lpips_score.cpu())

def main() -> None:
    parser = argparse.ArgumentParser(description="Search RMS-normalized eta for one image or every supported image in a folder.")
    parser.add_argument("--target-image", required=True, help="Target image file or folder used for RGC encoding and MSE.")
    parser.add_argument("--output-dir", required=True, help="Directory containing the aggregate results.csv.")
    parser.add_argument("--params", default=f"{_REPO_ROOT}/shared/params.yaml", help="Retina model parameters.")
    parser.add_argument("--diffusion-params", default=f"{_REPO_ROOT}/diffusion/diffusion_params.yaml", help="Diffusion defaults.")
    parser.add_argument("--model-id", default=None, help="Unconditional DDPM checkpoint override.")
    parser.add_argument("--backbone", choices=("diffusers", "dps"), default=None,
                        help="Unconditional prior to guide. Defaults to the diffusers model.")
    parser.add_argument("--dps-checkpoint", default=None,
                        help="Path to the published DPS .pt checkpoint, for --backbone dps.")
    parser.add_argument("--dps-preset", default=None,
                        help="DPS architecture preset, such as imagenet256 or ffhq256.")
    parser.add_argument("--image-size", type=int, default=None, help="Square reconstruction size override.")
    parser.add_argument("--include-cell-types", nargs="+", default=None, help="Optional RGC cell-type ablation.")
    parser.add_argument("--sigma-r", type=float, default=None, help="Fixed RGC rate uncertainty, recorded for reproducibility.")
    parser.add_argument("--etas", type=float, nargs="+", default=None,
                        help="Explicit guidance-strength grid. Defaults to the full "
                             "zero-plus-logarithmic grid from eta_grid().")
    parser.add_argument("--seed", type=int, default=0, help="Single shared random seed for every trajectory.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    target_root = Path(args.target_image)
    image_paths, folder_mode = collect_target_images(target_root)
    diffusion_params = u.read_params(args.diffusion_params)
    model_id = args.model_id or diffusion_params["model_id"]
    backbone = args.backbone or diffusion_params.get("backbone", "diffusers")
    dps_checkpoint = args.dps_checkpoint or diffusion_params.get("dps_checkpoint")
    dps_preset = args.dps_preset or diffusion_params.get("dps_preset", "imagenet256")
    image_size = args.image_size or diffusion_params["image_size"]
    sigma_r = args.sigma_r or diffusion_params["rate_noise_std"]
    if sigma_r <= 0:
        raise ValueError("--sigma-r must be positive.")
    include_cell_types = args.include_cell_types or diffusion_params.get("include_cell_types")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "reconstructions"
    if not folder_mode:
        image_dir.mkdir(parents=True, exist_ok=True)

    # An explicit grid lets a sweep bracket a known optimum instead of re-running
    # the full logarithmic scan; the per-combination eta selection downstream
    # still needs more than one value to choose between.
    etas = sorted(args.etas) if args.etas else eta_grid()
    saved_steps = timestep_grid()
    per_image_outputs = len(etas) * len(saved_steps)
    total_outputs = len(image_paths) * per_image_outputs
    print(f"Found {len(image_paths)} target image(s); folder mode: {folder_mode}.")
    print(f"Loading retina parameters from {args.params} and building {image_size}x{image_size} mosaics...")
    
    retina_params = u.read_params(args.params)
    encoder = StaticRGCEncoder.from_params(retina_params, (image_size, image_size), include_cell_types).to(args.device).eval()
    print(f"Included cell types: {', '.join(encoder.cell_type_names)}")
    
    if backbone == "dps":
        print(f"Loading DPS prior: {dps_checkpoint} (preset {dps_preset})")
    else:
        print(f"Loading unconditional DDPM prior: {model_id}")
    initial_likelihood = DeterministicRateLikelihood(encoder, rate_noise_std=sigma_r)
    reconstructor = DiffusionMAPReconstructor(
        initial_likelihood, model_id=model_id, device=args.device,
        backbone=backbone, dps_checkpoint=dps_checkpoint, dps_preset=dps_preset,
    )
    
    print("Setting up LPIPS scoring metric")
    lpips_metric = lpips.LPIPS(net="alex").to(args.device)
    lpips_metric.eval()
    
    csv_path = output_dir / "results.csv"
    if folder_mode:
        print("Folder mode: reconstructions and RGC-rate tensors will not be saved; only aggregate CSV rows are written.")
    else:
        print(f"Single-image mode: reconstructions and target RGC rates will be saved under {output_dir}.")
    print(f"Starting {len(image_paths) * len(etas)} RMS-normalized eta trajectories and collecting {total_outputs} " f"metric rows (fixed sigma_r={sigma_r:g}).")

    with csv_path.open("w", newline="") as csv_file:
        # "backbone" and "prior" are appended rather than inserted so that older
        # results.csv files (which lack them) still satisfy the column checks in
        # the analysis notebooks; absent values there mean the diffusers prior.
        writer = csv.DictWriter(csv_file, fieldnames=("source_image", "eta", "sigma_r", "T", "mse", "lpips", "final_rgc_loss", "seed", "backbone", "prior"))
        writer.writeheader()
        
        progress = tqdm(total=total_outputs, desc="Hyperparameter search", unit="image")

        for image_index, image_path in enumerate(image_paths, start=1):
            source_image = source_name(image_path, target_root, folder_mode)
            print(f"[{image_index}/{len(image_paths)}] Loading, resizing, and encoding {source_image}...")
            target_image = load_grayscale_image(str(image_path), image_size).to(args.device)
            with torch.no_grad():
                observed_rates = [rates.detach() for rates in encoder(target_image)]

            if not folder_mode:
                rates_path = output_dir / "target_rgc_rates.pt"
                torch.save(
                    {
                        "rates": [rates.cpu() for rates in observed_rates],
                        "cell_types": encoder.cell_type_names,
                        "image_size": image_size,
                        "source_image": source_image,
                    },
                    rates_path,
                )
                print(f"Saved target RGC observations to {rates_path}.")

            for eta in etas:
                for completed_steps, noisy_output, final_loss, mse, lpips_score in run_trajectory(reconstructor, observed_rates, target_image, lpips_metric, eta, set(saved_steps), args.seed):
                    
                    # store the image
                    os.makedirs(image_dir, exist_ok=True)
                    filename = f"{source_image.split('.')[0]}_reconstruction_{parameter_tag(eta, sigma_r, completed_steps)}.png"
                    save_grayscale_image(noisy_output, image_dir / filename)
                    
                    # store the output
                    writer.writerow(
                        {
                            "source_image": source_image,
                            "eta": eta,
                            "sigma_r": sigma_r,
                            "T": completed_steps,
                            "mse": mse,
                            "lpips": lpips_score,
                            "final_rgc_loss": final_loss,
                            "seed": args.seed,
                            "backbone": backbone,
                            "prior": dps_preset if backbone == "dps" else model_id,
                        }
                    )
                    csv_file.flush()
                    progress.update(1)
                    progress.set_postfix(image=image_index, eta=f"{eta:.2g}", T=completed_steps, mse=f"{mse:.4g}")
        progress.close()

    print(f"Search complete. Saved {total_outputs} rows to {csv_path}.")

if __name__ == "__main__":
    main()
