"""Compare guidance-strength arms using the calibration diagnostics themselves.

The guidance scale is the free parameter of the sampler, and the exchangeability
diagnostics are exactly the instrument for setting it: a correctly calibrated
sampler has ``S_YY / S_XX = 1`` and ``|S_XY| / S_{Y_j Y_k} = 1`` at every
frequency.  This script scores each arm of a sweep on those criteria, along with
reconstruction accuracy, and writes a comparison table and figure.

All arms are expected to share targets and seeds, so differences are attributable
to the guidance scale alone.
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
from diffusion.analysis import spectral_fillin as sf


def score_arm(path: Path, n_bins: int, n_boot: int) -> dict:
    """Run the diagnostics on one sweep arm and reduce them to comparable scalars.

    The headline score is the root-mean-square of ``log2(S_YY / S_XX)`` over the
    bins below the target content limit.  Using log2 makes the two directions of
    miscalibration symmetric, so a sampler twice as powerful as the truth and one
    half as powerful are penalised equally, which a raw ratio would not do.

    Args:
        path (Path): Path to the arm's bank ``.npz``.
        n_bins (int): Radial bins.
        n_boot (int): Bootstrap resamples.

    Returns:
        dict: Scalar scores plus the curves needed for the comparison figure.
    """
    bank = pl.load_bank(path)
    prepared = pl.prepare(bank, window="hann", n_bins=n_bins)
    nyquist = pl.nyquist_table(bank.spacings)
    finest = min(entry["nyquist_min"] for entry in nyquist.values())
    limit = (bank.config.get("bandwidth") or {}).get("content_limit_min", 0.5)

    coherence = pl.run_coherence(prepared, n_boot=n_boot, n_shuffles=4)
    fill = pl.run_fill(prepared, n_boot=n_boot)
    calibration = pl.run_calibration(prepared, n_boot=n_boot)

    f = prepared.radial.f_centers
    usable = (f <= limit) & np.isfinite(calibration["log2_ratio_power"])
    reliable = calibration["reliable"] & usable

    mean_y = bank.y.mean(axis=1)
    mse = float(np.mean((bank.x - mean_y) ** 2))
    single_mse = float(np.mean((bank.x[:, None] - bank.y) ** 2))
    flat_x = (bank.x - bank.x.mean(axis=(1, 2), keepdims=True)).reshape(bank.x.shape[0], -1)
    flat_y = (mean_y - mean_y.mean(axis=(1, 2), keepdims=True)).reshape(bank.x.shape[0], -1)
    correlation = float(np.mean(
        (flat_x * flat_y).sum(axis=1)
        / np.sqrt((flat_x**2).sum(axis=1) * (flat_y**2).sum(axis=1))
    ))

    return {
        "label": f"g={bank.config['guidance_scale']}, T={bank.config['steps']}",
        "guidance_scale": float(bank.config["guidance_scale"]),
        "steps": int(bank.config["steps"]),
        "n_images": bank.shape[0],
        "n_samples": bank.shape[1],
        "power_ratio_median": float(np.nanmedian(calibration["ratio_power"][usable])),
        "log2_power_rms": float(np.sqrt(np.nanmean(calibration["log2_ratio_power"][usable] ** 2))),
        "log2_power_max_abs": float(np.nanmax(np.abs(calibration["log2_ratio_power"][usable]))),
        "cross_ratio_median": (
            float(np.nanmedian(calibration["ratio_cross_raw"][reliable])) if reliable.any() else None
        ),
        "n_reliable_bins": int(reliable.sum()),
        "crossover_cycles_per_pixel": pl.crossover_frequency(f, fill["fill_fraction"]),
        "bits_single_sample": coherence["single"]["total_bits"],
        "bits_sample_mean": coherence["mean"]["total_bits"],
        "mse_sample_mean": mse,
        "mse_single_sample": single_mse,
        "pixel_correlation_mean": correlation,
        "final_guidance_loss_median": float(np.median(bank.losses[:, :, -1])),
        "error_share_observed_median": float(np.nanmedian(fill["observed_error_share"])),
        "error_share_target": fill["calibrated_error_share"],
        "_curves": {
            "f": f,
            "ratio_power": calibration["ratio_power"],
            "log2_ratio_power": calibration["log2_ratio_power"],
            "ratio_cross": calibration["ratio_cross_raw"],
            "reliable": calibration["reliable"],
            "fill_fraction": fill["fill_fraction"],
            "gamma2": coherence["single"]["gamma2"],
            "S_XX": coherence["single"]["S_XX"],
            "S_YY": coherence["single"]["S_YY"],
        },
        "_nyquist": nyquist,
        "_limit": limit,
        "_bank": bank,
        "_finest": finest,
    }


def main() -> None:
    """Score every sweep arm, write the comparison table, and plot the comparison."""
    parser = argparse.ArgumentParser(description="Compare guidance-strength sweep arms.")
    parser.add_argument("--banks", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arms = [score_arm(Path(p), args.n_bins, args.n_boot) for p in args.banks]
    arms.sort(key=lambda a: (a["guidance_scale"], a["steps"]))

    columns = ["label", "log2_power_rms", "power_ratio_median", "cross_ratio_median",
               "crossover_cycles_per_pixel", "bits_single_sample", "mse_sample_mean",
               "pixel_correlation_mean", "final_guidance_loss_median"]
    widths = [22, 14, 18, 17, 17, 18, 15, 15, 16]
    print("".join(name[:w - 1].rjust(w) for name, w in zip(columns, widths)))
    for arm in arms:
        cells = []
        for name, width in zip(columns, widths):
            value = arm[name]
            if isinstance(value, float):
                cells.append(f"{value:.4g}".rjust(width))
            else:
                cells.append(str(value).rjust(width))
        print("".join(cells))

    best = min(arms, key=lambda a: a["log2_power_rms"])
    print(f"\nBest calibrated arm by RMS log2 power ratio: {best['label']}")

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(arms)))
    for arm, color in zip(arms, colors):
        curves = arm["_curves"]
        f = curves["f"]
        axes[0, 0].plot(f, curves["ratio_power"], "o-", color=color, ms=3, label=arm["label"])
        axes[0, 1].plot(f, curves["gamma2"], "o-", color=color, ms=3, label=arm["label"])
        axes[1, 0].plot(f, curves["fill_fraction"], "o-", color=color, ms=3, label=arm["label"])
        mask = curves["reliable"]
        axes[1, 1].plot(f[mask], curves["ratio_cross"][mask], "o-", color=color, ms=3, label=arm["label"])

    axes[0, 0].axhline(1.0, color="k", ls="--", lw=1.4)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"$S_{YY}/S_{XX}$")
    axes[0, 0].set_title("Power ratio: 1 is calibrated")
    axes[0, 1].set_ylabel(r"$\gamma^2$")
    axes[0, 1].set_title("Single-sample coherence")
    axes[1, 0].set_ylabel("fill fraction")
    axes[1, 0].set_title("Fill-in")
    axes[1, 1].axhline(1.0, color="k", ls="--", lw=1.4)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel(r"$|S_{XY}|/S_{Y_jY_k}$")
    axes[1, 1].set_title("Cross ratio, reliable bins only")

    reference = arms[0]
    for axis in axes.ravel():
        pp.add_nyquist_lines(axis, reference["_nyquist"], label=False)
        pp.add_content_limit(axis, reference["_limit"], label=False)
        axis.set_xlabel("spatial frequency (cycles/pixel)")
        axis.legend(fontsize=6.5)
    figure.suptitle("Guidance-strength sweep: choosing the operating point with the calibration diagnostics")
    figure.tight_layout()
    figure.savefig(output_dir / "guidance_sweep_comparison.png", dpi=150)
    plt.close(figure)

    n_arms = len(arms)
    gallery, gaxes = plt.subplots(n_arms, 4, figsize=(10.0, 2.6 * n_arms))
    gaxes = np.atleast_2d(gaxes)
    for row, arm in enumerate(arms):
        bank = arm["_bank"]
        gaxes[row, 0].imshow(bank.x[0], cmap="gray", vmin=0, vmax=1)
        gaxes[row, 1].imshow(bank.y[0].mean(axis=0), cmap="gray", vmin=0, vmax=1)
        gaxes[row, 2].imshow(bank.y[0, 0], cmap="gray", vmin=0, vmax=1)
        gaxes[row, 3].imshow(bank.y[0].var(axis=0, ddof=1), cmap="magma")
        gaxes[row, 0].set_ylabel(arm["label"], fontsize=7)
        for axis in gaxes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    for column, title in enumerate(["target $X$", r"$\bar{Y}$", "$Y_1$", r"$\mathrm{Var}(Y|Z)$"]):
        gaxes[0, column].set_title(title, fontsize=9)
    gallery.suptitle("Same target and seeds at every guidance strength")
    gallery.tight_layout()
    gallery.savefig(output_dir / "guidance_sweep_gallery.png", dpi=150)
    plt.close(gallery)

    table = [{key: arm[key] for key in arm if not key.startswith("_")} for arm in arms]
    (output_dir / "guidance_sweep.json").write_text(json.dumps(
        {"arms": table, "best_by_log2_power_rms": best["label"],
         "criterion": ("RMS of log2(S_YY/S_XX) over bins below the target content limit; "
                       "log2 makes over- and under-expression symmetric.")},
        indent=2,
    ))
    print(f"Wrote {output_dir}/guidance_sweep.json and 2 figures.")


if __name__ == "__main__":
    main()
