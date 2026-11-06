"""Create static RGC-rate observations from a standard image file."""

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

import numpy as np
import torch
from PIL import Image

from diffusion.diffusion_retina import StaticRGCEncoder
from shared.sk_utils import Utils as u


def _load_grayscale_image(image_path: str, size: int) -> torch.Tensor:
    """Load, grayscale-convert, and resize an image for static RGC encoding.

    Args:
        image_path (str): Path to an image readable by Pillow, such as PNG or JPEG.
        size (int): Required image height and width.

    Returns:
        torch.Tensor: Float image tensor shaped (1, 1, size, size).
    """
    with Image.open(image_path) as source:
        grayscale = source.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.asarray(grayscale, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)

def main() -> None:
    parser = argparse.ArgumentParser(description="Encode one grayscale image as deterministic static RGC rates.")
    parser.add_argument("--image", required=True, help="Input image file, such as PNG or JPEG.")
    parser.add_argument("--params", default=f"{_REPO_ROOT}/shared/params.yaml")
    parser.add_argument("--diffusion-params", default=f"{_REPO_ROOT}/diffusion/diffusion_params.yaml",
                        help="Diffusion configuration containing an optional include_cell_types list.")
    parser.add_argument("--output", default="diffusion_rates.pt")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--include-cell-types", nargs="+", default=None,
                        help="Cell-type tags to encode; overrides diffusion.include_cell_types in --params.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading and resizing {args.image} to {args.image_size}x{args.image_size} grayscale...")
    image = _load_grayscale_image(args.image, args.image_size).to(args.device)
    print(f"Loading retina parameters from {args.params} and building hexagonal mosaics...")
    diffusion_params = u.read_params(args.diffusion_params)
    include_cell_types = args.include_cell_types or diffusion_params.get("include_cell_types")
    encoder = StaticRGCEncoder.from_params(
        u.read_params(args.params), (args.image_size, args.image_size), include_cell_types
    ).to(args.device).eval()
    print(f"Included cell types: {', '.join(encoder.cell_type_names)}")
    print(f"Encoding deterministic RGC rates on {args.device}...")
    with torch.no_grad():
        rates = [rate.cpu() for rate in encoder(image)]
    print(f"Saving rates for {len(rates)} mosaics to {args.output}...")
    torch.save(
        {
            "rates": rates,
            "cell_types": encoder.cell_type_names,
            "image_size": args.image_size,
        },
        args.output,
    )
    print("Static RGC encoding complete.")


if __name__ == "__main__":
    main()
