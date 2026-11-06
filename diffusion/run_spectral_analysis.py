"""Run the full spectral coherence and fill-in analysis on a saved bank.

This is the headless counterpart of the analysis notebook.  Both call the same
functions in ``analysis.pipeline`` and ``analysis.plots``, so the notebook and
the script cannot drift apart.

Usage::

    python run_spectral_analysis.py --bank results_prior-vs-likelihood/banks/run.npz \
        --output-dir results_prior-vs-likelihood/main
"""

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
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffusion.analysis import pipeline as pl
from diffusion.analysis import plots as pp


def main() -> None:
    """Run every analysis stage and write arrays, figures, and a summary."""
    parser = argparse.ArgumentParser(description="Analyse a reconstruction bank.")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", default="hann", choices=("hann", "tukey", "none"))
    parser.add_argument("--tukey-alpha", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--n-orient", type=int, default=8)
    parser.add_argument("--n-bins-oriented", type=int, default=16)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--n-shuffles", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--microns-per-pixel", type=float, default=None,
                        help="Retinal microns per image pixel; enables the cycles/degree axis.")
    parser.add_argument("--microns-per-degree", type=float, default=None,
                        help="Retinal microns per degree; enables the cycles/degree axis.")
    parser.add_argument("--no-cluster-bootstrap", action="store_true",
                        help="Resample tiles rather than source scenes, which understates the CIs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    bank = pl.load_bank(args.bank)
    print(bank.describe())
    prepared = pl.prepare(
        bank, window=args.window, alpha=args.tukey_alpha, n_bins=args.n_bins,
        n_orient=args.n_orient, n_bins_oriented=args.n_bins_oriented,
    )
    nyquist = pl.nyquist_table(bank.spacings)
    cluster = not args.no_cluster_bootstrap
    content_limit = (bank.config.get("bandwidth") or {}).get("content_limit_min", 0.5)
    scale = {"microns_per_pixel": args.microns_per_pixel,
             "microns_per_degree": args.microns_per_degree,
             "content_limit": content_limit}
    print(f"Target content limit: {content_limit:.4f} cycles/pixel.")

    print("Analysis 1: coherence and information...")
    coherence = pl.run_coherence(prepared, args.n_boot, args.n_shuffles, args.seed, cluster)
    print("Analysis 2: fill-in...")
    fill = pl.run_fill(prepared, args.n_boot, args.seed, cluster)
    print("Calibration diagnostics...")
    calibration = pl.run_calibration(prepared, args.n_boot, args.seed, cluster)
    print("Orientation-resolved analysis...")
    oriented = pl.run_oriented(prepared)

    # Split per-image scores at the lowest mosaic Nyquist (the coarsest-sampling
    # mosaic), the conservative frequency below which no mosaic aliases.
    nyquist_low = min(entry["nyquist_min"] for entry in nyquist.values())
    per_image = pl.run_per_image(prepared, cutoff=nyquist_low)

    print("Figures...")
    figures = {
        "spectra": pp.plot_spectra(coherence, nyquist, **scale),
        "coherence": pp.plot_coherence(coherence, nyquist, **scale),
        "information": pp.plot_information(coherence, nyquist, **scale),
        "fill_curves": pp.plot_fill_curves(fill, nyquist, **scale),
        "error_decomposition": pp.plot_error_decomposition(fill, nyquist, **scale),
        "calibration": pp.plot_calibration(calibration, nyquist, **scale),
        "oriented": pp.plot_oriented(oriented, bank.spacings),
        "pixel_maps": pp.plot_pixel_maps(fill),
        "per_image_scores": pp.plot_per_image_scores(per_image),
        "loss_traces": pp.plot_loss_traces(bank),
    }
    order = per_image["order"]
    examples = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    figures["examples"] = pp.plot_examples(bank, fill, examples)
    for name, figure in figures.items():
        figure.savefig(figure_dir / f"{name}.png", dpi=150)
        plt.close(figure)

    print("Saving arrays and summary...")
    arrays = pl.collect_arrays(
        coherence=coherence, fill=fill, calibration=calibration,
        oriented=oriented, per_image=per_image,
    )
    arrays["radial/f_centers"] = prepared.radial.f_centers
    arrays["radial/f_edges"] = prepared.radial.f_edges
    arrays["radial/dof"] = prepared.radial.dof
    arrays["oriented/theta_centers"] = prepared.oriented.theta_centers
    np.savez_compressed(output_dir / "spectral_fillin_results.npz", **arrays)

    summary = pl.build_summary(bank, prepared, coherence, fill, calibration, per_image, nyquist)
    summary["preprocessing"]["window"] = args.window
    summary["preprocessing"]["tukey_alpha"] = args.tukey_alpha
    summary["preprocessing"]["n_radial_bins"] = args.n_bins
    summary["preprocessing"]["bootstrap"] = "scene-clustered" if cluster else "per-tile"
    summary["preprocessing"]["n_bootstrap"] = args.n_boot
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote {output_dir}/summary.json and {len(figures)} figures.")
    print(json.dumps(
        {
            "crossover_cycles_per_pixel": summary["fill_in"]["crossover_frequency_cycles_per_pixel"],
            "bits_per_image_single": summary["information"]["single_sample_total_bits_per_image"],
            "power_ratio_median": summary["calibration"]["power_ratio_median"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
