"""Build a bank of repeated RGC-guided reconstructions for the spectral analysis.

The analysis in ``analysis/spectral_fillin.py`` needs many reconstructions of the
same latent, drawn with different seeds, so that the within-Z variance can be
separated from the across-image variance.  ``diffusion_reconstruct.py`` produces
one image from one seed and writes a PNG; this driver wraps it without modifying
it, and writes arrays instead.

It imports the model code and never changes it.  All model-dependent work happens
here; everything downstream reads only the saved ``.npz``.

Output layout::

    X       (N, H, W)      float32, targets in [0, 1]
    Y       (N, K, H, W)   float32, reconstructions in [0, 1]
    seeds   (N, K)         int64, the seed used for each reconstruction
    losses  (N, K, steps)  float32, the per-timestep guidance loss trace
    rates__<cell_type>     (N, n_cells) float32, the latent Z per mosaic
    config                 JSON string of every resolved setting
    mosaic_spacings        JSON string mapping cell type to lattice spacing

Normalization.  Targets and reconstructions are both stored as raw grayscale in
[0, 1], with no per-image rescaling.  This matters: rescaling each image to a
common variance would force ``S_XX = S_YY`` and silently destroy the calibration
diagnostic the analysis depends on.  The per-image mean subtraction that the
spectral analysis needs happens there, not here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from diffusion_map import DeterministicRateLikelihood
from diffusion_reconstruct import DiffusionMAPReconstructor
from diffusion_retina import StaticRGCEncoder
from sk_utils import Utils as u

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def tile_source_image(path: Path, size: int) -> list[np.ndarray]:
    """Cut a high-resolution image into non-overlapping grayscale tiles.

    Tiling rather than resizing is deliberate.  Downsampling a large photograph
    to 256 pixels destroys exactly the high spatial frequencies this analysis is
    about: the fill fraction above the resampling Nyquist would be trivially one,
    for reasons having nothing to do with the retina or the sampler.  Native-
    resolution tiles keep the full bandwidth of the source.

    Tiles are taken from a centred grid so that the discarded remainder is split
    evenly between the edges.

    Args:
        path (Path): Source image path.
        size (int): Tile height and width in pixels.

    Returns:
        list[numpy.ndarray]: Tiles shaped ``(size, size)`` with values in [0, 1].
    """
    with Image.open(path) as source:
        grayscale = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
    height, width = grayscale.shape
    n_rows, n_cols = height // size, width // size
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f"{path} is smaller than one {size}x{size} tile.")
    top, left = (height - n_rows * size) // 2, (width - n_cols * size) // 2
    return [
        np.ascontiguousarray(grayscale[top + r * size : top + (r + 1) * size,
                                       left + c * size : left + (c + 1) * size])
        for r in range(n_rows)
        for c in range(n_cols)
    ]


def center_crop_resize(path: Path, size: int) -> np.ndarray:
    """Centre-crop an image to a square and resample it to the model's input size.

    This is the right loader for a large collection of small photographs, where
    each file is one independent scene.  It costs bandwidth: a source whose
    shorter side is ``w`` pixels carries no content above ``w / (2 * size)``
    cycles per pixel once resampled, and that limit should be recorded and drawn
    on the figures so the band above it is never read as a property of the retina
    or the sampler.

    Args:
        path (Path): Source image path.
        size (int): Output height and width in pixels.

    Returns:
        numpy.ndarray: Image shaped ``(size, size)`` with values in [0, 1].
    """
    with Image.open(path) as source:
        grayscale = source.convert("L")
        width, height = grayscale.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        square = grayscale.crop((left, top, left + side, top + side))
        resampled = square.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(resampled, dtype=np.float32) / 255.0


def source_content_limit(path: Path, size: int) -> float:
    """Return the highest frequency a resampled source can still carry.

    Args:
        path (Path): Source image path.
        size (int): Output height and width in pixels.

    Returns:
        float: Content limit in cycles per pixel of the output grid, capped at 0.5.
    """
    with Image.open(path) as source:
        width, height = source.size
    return float(min(0.5, min(width, height) / (2.0 * size)))


def build_target_bank(
    sources: list[Path], size: int, n_images: int | None, seed: int, mode: str = "tile"
) -> tuple[np.ndarray, list[str], dict]:
    """Assemble a deterministic target bank from several sources or source groups.

    Two modes serve two different situations:

    * ``tile`` cuts each high-resolution source into non-overlapping native
      resolution tiles.  Bandwidth is preserved in full, but tiles from one
      photograph are not independent samples, so the label records the source and
      the analysis clusters its bootstrap on it.
    * ``resize`` treats every file as one independent scene, centre-cropped and
      resampled.  Independence is real, at the cost of whatever bandwidth the
      source lacked.

    In both modes entries are interleaved across groups before truncation, so a
    bank smaller than the available pool still draws evenly from every group
    rather than exhausting the first one.

    Args:
        sources (list[Path]): Source images, or directories in ``resize`` mode.
        size (int): Target height and width in pixels.
        n_images (int | None): Bank size, or ``None`` to use everything available.
        seed (int): Seed for the fixed shuffle applied within each group.
        mode (str): ``"tile"`` or ``"resize"``.

    Returns:
        tuple[numpy.ndarray, list[str], dict]: Targets shaped ``(N, size, size)``,
        a provenance label per target, and bandwidth metadata.
    """
    rng = np.random.default_rng(seed)
    groups: list[list[tuple[Any, str]]] = []

    if mode == "tile":
        for path in sources:
            tiles = tile_source_image(path, size)
            order = rng.permutation(len(tiles))
            # The colon marks tiles that share a source, which the analysis reads
            # as a cluster label for an honest bootstrap.
            groups.append([(tiles[i], f"{path.stem}:tile{i:03d}") for i in order])
    elif mode == "resize":
        for root in sources:
            files = (
                sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
                if root.is_dir() else [root]
            )
            order = rng.permutation(len(files))
            # No colon: every file is its own scene, so the cluster bootstrap
            # correctly degenerates to an ordinary per-image bootstrap.
            groups.append([(files[i], f"{root.name}/{files[i].stem}") for i in order])
    else:
        raise ValueError(f"Unsupported target mode: {mode}")

    if not any(groups):
        raise ValueError("No usable source images were found.")

    interleaved = []
    for position in range(max(len(group) for group in groups)):
        for group in groups:
            if position < len(group):
                interleaved.append(group[position])
    if n_images is not None:
        if n_images > len(interleaved):
            raise ValueError(f"Requested {n_images} targets but only {len(interleaved)} are available.")
        interleaved = interleaved[:n_images]

    if mode == "tile":
        images = np.stack([entry for entry, _ in interleaved], axis=0)
        limits = [0.5] * len(interleaved)
    else:
        images = np.stack([center_crop_resize(entry, size) for entry, _ in interleaved], axis=0)
        limits = [source_content_limit(entry, size) for entry, _ in interleaved]

    bandwidth = {
        "mode": mode,
        "content_limit_min": float(np.min(limits)),
        "content_limit_median": float(np.median(limits)),
        "content_limit_max": float(np.max(limits)),
        "note": (
            "Highest spatial frequency the targets can carry, in cycles per pixel. "
            "In resize mode this is set by the source resolution, and bins above it "
            "say nothing about the retina or the sampler."
        ),
    }
    return images, [label for _, label in interleaved], bandwidth


def mosaic_spacings(encoder: StaticRGCEncoder) -> dict[str, float]:
    """Read the hexagonal lattice spacing actually used by each mosaic.

    The spacing is not the ``base_spacing`` key in the YAML, which the tiling
    code ignores.  ``_tile_cells`` builds the lattice with
    ``s = spatial.center_size / tiling_config.coverage_factor``, so that is what
    determines the mosaic's Nyquist frequency and what is recorded here.

    Args:
        encoder (StaticRGCEncoder): Encoder whose mosaics were already built.

    Returns:
        dict[str, float]: Lattice spacing in pixels, keyed by cell type.
    """
    spacings = {}
    for name, mosaic in zip(encoder.cell_type_names, encoder.mosaics):
        params = mosaic.params
        coverage = float(params["tiling_config"].get("coverage_factor", 1.0))
        spacings[name] = float(params["spatial"]["center_size"]) / coverage
    return spacings


def sample_one_image(
    reconstructor: DiffusionMAPReconstructor,
    encoder: StaticRGCEncoder,
    target: np.ndarray,
    seeds: list[int],
    steps: int,
    guidance_scale: float,
    guidance_normalization: str,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Encode one target and draw K reconstructions of it from different seeds.

    Args:
        reconstructor (DiffusionMAPReconstructor): Guided sampler, already loaded.
        encoder (StaticRGCEncoder): Static RGC encoder defining the latent.
        target (numpy.ndarray): Target image shaped ``(H, W)`` with values in [0, 1].
        seeds (list[int]): One seed per reconstruction.
        steps (int): Reverse-diffusion steps.
        guidance_scale (float): Likelihood-gradient magnitude.
        guidance_normalization (str): ``none`` or ``rms``.
        device (torch.device): Device for the sampler.

    Returns:
        dict[str, numpy.ndarray]: Keys ``samples``, ``losses``, and one
        ``rates__<cell_type>`` entry per mosaic.
    """
    image = torch.from_numpy(target)[None, None].to(device)
    with torch.no_grad():
        rates = [rate.detach() for rate in encoder(image)]

    samples, losses = [], []
    for seed in seeds:
        reconstruction, trace = reconstructor.reconstruct(
            rates, steps, guidance_scale, guidance_normalization, int(seed)
        )
        samples.append(reconstruction.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32))
        losses.append(np.asarray(trace, dtype=np.float32))

    out = {
        "samples": np.stack(samples, axis=0),
        "losses": np.stack(losses, axis=0),
    }
    for name, rate in zip(encoder.cell_type_names, rates):
        out[f"rates__{name}"] = rate.detach().cpu().numpy().astype(np.float32).squeeze(0)
    return out


def main() -> None:
    """Build and save a reconstruction bank, resuming from any existing shards."""
    parser = argparse.ArgumentParser(description="Sample K RGC-guided reconstructions for each of N targets.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--diffusion-params", default="diffusion_params.yaml")
    parser.add_argument("--sources", nargs="+", required=True,
                        help="Source images, or directories, depending on --target-mode.")
    parser.add_argument("--target-mode", choices=("tile", "resize"), default="tile",
                        help="tile: cut high-resolution sources into native tiles. "
                             "resize: treat each file as one scene, centre-cropped and resampled.")
    parser.add_argument("--output", required=True, help="Destination .npz for the consolidated bank.")
    parser.add_argument("--shard-dir", default=None,
                        help="Directory for per-image shards; defaults to <output>.shards.")
    parser.add_argument("--n-images", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20260101)
    parser.add_argument("--tile-seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--guidance-normalization", choices=("none", "rms"), default=None)
    parser.add_argument("--rate-noise-std", type=float, default=None)
    parser.add_argument("--include-cell-types", nargs="+", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    diffusion_params = u.read_params(args.diffusion_params)
    model_id = args.model_id or diffusion_params["model_id"]
    steps = args.steps or diffusion_params["num_inference_steps"]
    image_size = args.image_size or diffusion_params["image_size"]
    guidance_scale = args.guidance_scale or diffusion_params["guidance_scale"]
    guidance_normalization = args.guidance_normalization or diffusion_params.get("guidance_normalization", "rms")
    rate_noise_std = args.rate_noise_std or diffusion_params["rate_noise_std"]
    include_cell_types = args.include_cell_types or diffusion_params.get("include_cell_types")

    sources: list[Path] = []
    for entry in args.sources:
        path = Path(entry)
        if args.target_mode == "tile" and path.is_dir():
            sources.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        else:
            sources.append(path)
    if not sources:
        raise ValueError("No source images were found.")

    print(f"Building {image_size}x{image_size} targets from {len(sources)} source(s) "
          f"in {args.target_mode} mode...")
    targets, labels, bandwidth = build_target_bank(
        sources, image_size, args.n_images, args.tile_seed, args.target_mode
    )
    print(f"Target bank: {targets.shape[0]} targets.")
    print(f"Target content limit: {bandwidth['content_limit_min']:.4f} to "
          f"{bandwidth['content_limit_max']:.4f} cycles/pixel "
          f"(median {bandwidth['content_limit_median']:.4f}).")

    print(f"Building static RGC encoder from {args.params}...")
    params = u.read_params(args.params)
    encoder = StaticRGCEncoder.from_params(params, (image_size, image_size), include_cell_types)
    encoder = encoder.to(args.device).eval()
    spacings = mosaic_spacings(encoder)
    print("Cell types and lattice spacings (pixels):")
    for name, spacing in spacings.items():
        print(f"  {name}: {spacing:.4f}")

    print(f"Loading DDPM prior {model_id} onto {args.device}...")
    likelihood = DeterministicRateLikelihood(encoder, rate_noise_std)
    reconstructor = DiffusionMAPReconstructor(likelihood, model_id, args.device)
    device = torch.device(args.device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(args.shard_dir) if args.shard_dir else output.with_suffix(".shards")
    shard_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_id": model_id,
        "steps": steps,
        "image_size": image_size,
        "guidance_scale": guidance_scale,
        "guidance_normalization": guidance_normalization,
        "rate_noise_std": rate_noise_std,
        "cell_types": encoder.cell_type_names,
        "n_images": int(targets.shape[0]),
        "n_samples": int(args.n_samples),
        "base_seed": int(args.base_seed),
        "tile_seed": int(args.tile_seed),
        "sources": [str(p) for p in sources],
        "target_mode": args.target_mode,
        "bandwidth": bandwidth,
        "labels": labels,
        "params_file": args.params,
        "diffusion_params_file": args.diffusion_params,
        "device": args.device,
    }

    seeds = np.asarray(
        [[args.base_seed + n * args.n_samples + k for k in range(args.n_samples)]
         for n in range(targets.shape[0])],
        dtype=np.int64,
    )

    start = time.time()
    completed = 0
    for n in range(targets.shape[0]):
        shard = shard_dir / f"image_{n:04d}.npz"
        if shard.exists():
            print(f"[{n + 1}/{targets.shape[0]}] shard exists, skipping.")
            continue
        image_start = time.time()
        result = sample_one_image(
            reconstructor, encoder, targets[n], list(seeds[n]), steps,
            guidance_scale, guidance_normalization, device,
        )
        np.savez_compressed(shard, target=targets[n], seeds=seeds[n], label=labels[n], **result)
        completed += 1
        elapsed = time.time() - image_start
        per_sample = elapsed / args.n_samples
        remaining = (targets.shape[0] - n - 1) * elapsed
        print(
            f"[{n + 1}/{targets.shape[0]}] {labels[n]}  {elapsed:.1f}s "
            f"({per_sample:.1f}s/sample)  eta {remaining / 60:.1f} min"
        )

    print("Consolidating shards...")
    shards = [np.load(shard_dir / f"image_{n:04d}.npz") for n in range(targets.shape[0])]
    bundle = {
        "X": np.stack([s["target"] for s in shards]).astype(np.float32),
        "Y": np.stack([s["samples"] for s in shards]).astype(np.float32),
        "seeds": np.stack([s["seeds"] for s in shards]).astype(np.int64),
        "losses": np.stack([s["losses"] for s in shards]).astype(np.float32),
        "labels": np.asarray(labels),
        "config": json.dumps(config, indent=2),
        "mosaic_spacings": json.dumps(spacings, indent=2),
    }
    for name in encoder.cell_type_names:
        key = f"rates__{name}"
        bundle[key] = np.stack([s[key] for s in shards]).astype(np.float32)
    np.savez_compressed(output, **bundle)

    total = time.time() - start
    print(f"Wrote {output}  X={bundle['X'].shape}  Y={bundle['Y'].shape}")
    if completed:
        print(f"Sampled {completed} image(s) in {total / 60:.1f} min "
              f"({total / (completed * args.n_samples):.1f}s per reconstruction).")


if __name__ == "__main__":
    main()
