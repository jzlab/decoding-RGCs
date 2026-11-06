"""Orchestration for the RGC spectral coherence and fill-in analysis.

This module turns a saved reconstruction bank into every curve, interval, and
scalar the report needs.  It imports only :mod:`analysis.spectral_fillin` and
numpy, never the model code, so it runs anywhere the arrays can be read.

The stages are deliberately separate rather than one call, so that a notebook can
run them one at a time and inspect what each produces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import spectral_fillin as sf

__all__ = [
    "Bank",
    "load_bank",
    "Prepared",
    "prepare",
    "nyquist_table",
    "run_coherence",
    "run_fill",
    "run_calibration",
    "run_oriented",
    "run_per_image",
    "crossover_frequency",
    "collect_arrays",
    "build_summary",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class Bank:
    """A saved bank of targets and their repeated reconstructions.

    Attributes:
        x (numpy.ndarray): Targets shaped ``(N, H, W)`` in [0, 1].
        y (numpy.ndarray): Reconstructions shaped ``(N, K, H, W)`` in [0, 1].
        seeds (numpy.ndarray): Seed used for each reconstruction, ``(N, K)``.
        losses (numpy.ndarray): Guidance loss trace, ``(N, K, steps)``.
        labels (numpy.ndarray): Provenance label per target.
        config (dict): Every resolved sampling setting.
        spacings (dict): Mosaic lattice spacing in pixels, keyed by cell type.
        rates (dict): Latent firing rates per cell type, ``(N, n_cells)``.
        path (Path): Where the bank was read from.
    """

    x: np.ndarray
    y: np.ndarray
    seeds: np.ndarray
    losses: np.ndarray
    labels: np.ndarray
    config: dict
    spacings: dict
    rates: dict
    path: Path

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """tuple: ``(N, K, H, W)`` for this bank."""
        return (self.x.shape[0], self.y.shape[1], self.x.shape[1], self.x.shape[2])

    @property
    def scenes(self) -> np.ndarray:
        """numpy.ndarray: Source scene for each target, for the cluster bootstrap.

        Tiles cut from the same photograph are not independent samples, so the
        part of the label before the colon is used as a cluster label.
        """
        return np.asarray([str(label).split(":")[0] for label in self.labels])

    def describe(self) -> str:
        """Return a one-line human-readable description of the bank.

        Returns:
            str: Shape, scene count, and the sampler settings that matter.
        """
        n, k, h, w = self.shape
        return (
            f"{self.path.name}: N={n} targets ({len(set(self.scenes))} scenes), K={k} samples, "
            f"{h}x{w}, {self.config.get('steps')} steps, "
            f"guidance={self.config.get('guidance_scale')} "
            f"({self.config.get('guidance_normalization')}), "
            f"cell types={', '.join(self.config.get('cell_types', []))}"
        )


def load_bank(path: str | Path) -> Bank:
    """Read a reconstruction bank written by ``diffusion_sample_bank.py``.

    Args:
        path (str | Path): Path to the consolidated ``.npz``.

    Returns:
        Bank: Loaded arrays and metadata.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    rates = {key[len("rates__"):]: data[key] for key in data.files if key.startswith("rates__")}
    return Bank(
        x=data["X"].astype(np.float64),
        y=data["Y"].astype(np.float64),
        seeds=data["seeds"],
        losses=data["losses"],
        labels=data["labels"],
        config=json.loads(str(data["config"])),
        spacings=json.loads(str(data["mosaic_spacings"])),
        rates=rates,
        path=path,
    )


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


@dataclass
class Prepared:
    """Windowed spectra and bin assignments shared by every analysis stage.

    Attributes:
        bank (Bank): The bank these spectra came from.
        setup (sf.SpectralSetup): FFT geometry, window, and mode weights.
        radial (sf.Binning): Radial bin assignment.
        oriented (sf.Binning): Radius-by-orientation bin assignment.
        x_hat (numpy.ndarray): Target spectra, ``(N, H, W)`` complex.
        y_hat (numpy.ndarray): Reconstruction spectra, ``(N, K, H, W)`` complex.
    """

    bank: Bank
    setup: sf.SpectralSetup
    radial: sf.Binning
    oriented: sf.Binning
    x_hat: np.ndarray
    y_hat: np.ndarray

    @property
    def n_images(self) -> int:
        """int: Number of targets."""
        return int(self.x_hat.shape[0])

    @property
    def n_samples(self) -> int:
        """int: Number of reconstructions per target."""
        return int(self.y_hat.shape[1])


def prepare(
    bank: Bank,
    window: str = "hann",
    alpha: float = 0.5,
    n_bins: int = 32,
    n_orient: int = 8,
    n_bins_oriented: int = 16,
    f_max: float = 0.5,
) -> Prepared:
    """Window and transform the whole bank once, identically for X and every Y.

    Args:
        bank (Bank): Loaded reconstruction bank.
        window (str): Window kind, ``"hann"``, ``"tukey"``, or ``"none"``.
        alpha (float): Tukey taper fraction, ignored for other windows.
        n_bins (int): Radial bins for the one-dimensional curves.
        n_orient (int): Orientation bins for the oriented analysis.
        n_bins_oriented (int): Radial bins for the oriented analysis.
        f_max (float): Upper radial frequency, in cycles/pixel.

    Returns:
        Prepared: Spectra and binnings ready for every stage.
    """
    n, k, height, width = bank.shape
    setup = sf.make_setup((height, width), window, alpha)
    x_hat = sf.preprocess(bank.x, setup)
    y_hat = sf.preprocess(bank.y.reshape(-1, height, width), setup).reshape(n, k, height, width)
    return Prepared(
        bank=bank,
        setup=setup,
        radial=sf.radial_binning(setup, n_bins, f_max),
        oriented=sf.radial_orientation_binning(setup, n_bins_oriented, n_orient, f_max),
        x_hat=x_hat,
        y_hat=y_hat,
    )


def nyquist_table(spacings: dict[str, float]) -> dict[str, dict[str, float]]:
    """Compute the Nyquist frequencies of every mosaic from its lattice spacing.

    Each mosaic has two Nyquist frequencies, not one, because its lattice is
    hexagonal: the first Brillouin zone is a hexagon, so the highest resolvable
    frequency depends on direction.  The inradius is the conservative scalar for a
    radially averaged plot, since below it no orientation aliases.

    Args:
        spacings (dict[str, float]): Lattice spacing in pixels, keyed by cell type.

    Returns:
        dict[str, dict[str, float]]: Per cell type, the spacing and the inradius
        and circumradius of its Brillouin zone, in cycles/pixel.
    """
    table = {}
    for name, spacing in spacings.items():
        inradius, circumradius = sf.hex_nyquist_bounds(float(spacing))
        table[name] = {
            "spacing_pixels": float(spacing),
            "nyquist_min": inradius,
            "nyquist_max": circumradius,
        }
    return table


# ---------------------------------------------------------------------------
# Analysis 1: coherence and the information lower bound
# ---------------------------------------------------------------------------


def run_coherence(
    prepared: Prepared,
    n_boot: int = 400,
    n_shuffles: int = 8,
    seed: int = 0,
    cluster: bool = True,
) -> dict[str, Any]:
    """Run all three coherence variants with shuffle nulls and bootstrap intervals.

    The three variants answer different questions:

    * ``single``: X against one reconstruction.  This is what a user of the
      system actually gets, so it is the operationally meaningful number.
    * ``mean``: X against the average of the K reconstructions.  This estimates
      how much the latent determines, with the sampler's own noise averaged away.
      The naive version is biased **downward** at finite K; the split-half
      debiased version is reported alongside and should be preferred.
    * ``sample_pair``: one reconstruction against another.  This measures only
      what the latent pins down, with no reference to the truth, and equals
      ``(1 - F)^2`` where F is the fill fraction.

    Args:
        prepared (Prepared): Spectra and binnings.
        n_boot (int): Bootstrap resamples.
        n_shuffles (int): Derangements used for the shuffle null.
        seed (int): Base seed for nulls and bootstraps.
        cluster (bool): Whether to bootstrap over source scenes rather than tiles.

    Returns:
        dict[str, Any]: Per-variant curves, nulls, corrected coherence, bits, and
        bootstrap intervals, plus the total information in bits per image.
    """
    binning, setup = prepared.radial, prepared.setup
    groups = prepared.bank.scenes if cluster else None
    x_hat, y_hat = prepared.x_hat, prepared.y_hat

    variants = {
        "single": sf.per_image_bin_sums(x_hat, y_hat[:, 0], binning, setup),
        "mean": sf.per_image_bin_sums(x_hat, y_hat.mean(axis=1), binning, setup),
        "sample_pair": sf.sample_pair_spectra(y_hat, binning, setup),
    }

    results: dict[str, Any] = {"binning": binning, "f_centers": binning.f_centers}
    for name, spectra in variants.items():
        reduced = sf.reduce_bin_sums(spectra)
        if name == "sample_pair":
            null = {"gamma2_null": np.zeros(binning.n_bins), "bits_null": np.zeros(binning.n_bins)}
        else:
            partner = y_hat[:, 0] if name == "single" else y_hat.mean(axis=1)
            null = sf.shuffle_null_coherence(x_hat, partner, binning, setup, n_shuffles, seed + 1)
        corrected = sf.correct_coherence(reduced["gamma2"], null["gamma2_null"])
        ci = sf.bootstrap_ci(
            lambda idx, s=spectra: sf.reduce_bin_sums(s, idx)["gamma2"],
            n_images=spectra.n_images, n_boot=n_boot, seed=seed + 2, groups=groups,
        )
        results[name] = {
            "S_XX": reduced["S_XX"],
            "S_YY": reduced["S_YY"],
            "S_XY": reduced["S_XY"],
            "gamma2": reduced["gamma2"],
            "gamma2_null": null["gamma2_null"],
            "gamma2_corrected": corrected,
            "bits_per_dof": reduced["bits_per_dof"],
            "bits": reduced["bits"],
            "bits_corrected": sf.information_bits(corrected, binning.dof),
            "ci_lo": ci["lo"],
            "ci_hi": ci["hi"],
            "total_bits": float(np.nansum(reduced["bits"])),
            "total_bits_corrected": float(np.nansum(sf.information_bits(corrected, binning.dof))),
        }

    if prepared.n_samples >= 4:
        split = sf.split_half_mean_spectra(x_hat, y_hat, binning, setup, seed=seed + 3)
        debiased = sf.debiased_mean_coherence(split)
        ci = sf.bootstrap_ci(
            lambda idx: sf.debiased_mean_coherence(split, idx)["gamma2_debiased"],
            n_images=prepared.n_images, n_boot=n_boot, seed=seed + 4, groups=groups,
        )
        results["mean_debiased"] = {
            "gamma2": debiased["gamma2_debiased"],
            "gamma2_naive": debiased["gamma2_naive"],
            "bits": debiased["bits_debiased"],
            "ci_lo": ci["lo"],
            "ci_hi": ci["hi"],
            "total_bits": float(np.nansum(debiased["bits_debiased"])),
        }
    else:
        results["mean_debiased"] = None
    return results


# ---------------------------------------------------------------------------
# Analysis 2: fill-in
# ---------------------------------------------------------------------------


def run_fill(prepared: Prepared, n_boot: int = 400, seed: int = 0, cluster: bool = True) -> dict[str, Any]:
    """Compute fill-in curves, the two variance identities, and pixel-space maps.

    Args:
        prepared (Prepared): Spectra and binnings.
        n_boot (int): Bootstrap resamples.
        seed (int): Base seed for the bootstrap.
        cluster (bool): Whether to bootstrap over source scenes rather than tiles.

    Returns:
        dict[str, Any]: Fill curves with intervals, the law-of-total-variance and
        error-decomposition checks, and the pixel-space variance maps.
    """
    binning, setup = prepared.radial, prepared.setup
    groups = prepared.bank.scenes if cluster else None
    acc = sf.accumulate_fill(prepared.x_hat, prepared.y_hat, binning, setup)
    curves = sf.fill_curves(acc)
    ci = sf.bootstrap_ci(
        lambda idx: sf.fill_curves(acc, idx)["fill_fraction"],
        n_images=acc.n_images, n_boot=n_boot, seed=seed + 5, groups=groups,
    )
    total = sf.total_variance_check(acc)
    decomposition = sf.error_decomposition(acc)
    power = curves["S_YY"]
    return {
        "binning": binning,
        "f_centers": binning.f_centers,
        "accumulation": acc,
        **curves,
        "fill_ci_lo": ci["lo"],
        "fill_ci_hi": ci["hi"],
        "total_variance": total,
        "error_decomposition": decomposition,
        "total_variance_max_relative_residual": float(np.max(np.abs(total["residual"]) / power)),
        "error_max_relative_residual": float(np.max(np.abs(decomposition["residual"]) / power)),
        "calibrated_error_share": (prepared.n_samples - 1) / (2.0 * prepared.n_samples),
        "observed_error_share": decomposition["fill_term"] / decomposition["lhs"],
        "pixel_maps": sf.pixel_variance_maps(prepared.bank.y),
    }


def run_calibration(prepared: Prepared, n_boot: int = 400, seed: int = 0,
                    cluster: bool = True, min_shared: float = 0.05) -> dict[str, Any]:
    """Run the two exchangeability diagnostics with bootstrap intervals.

    Args:
        prepared (Prepared): Spectra and binnings.
        n_boot (int): Bootstrap resamples.
        seed (int): Base seed for the bootstrap.
        cluster (bool): Whether to bootstrap over source scenes rather than tiles.
        min_shared (float): Smallest shared fraction at which the cross ratio is read.

    Returns:
        dict[str, Any]: Both ratios, their intervals, and the reliability mask.
    """
    binning, setup = prepared.radial, prepared.setup
    groups = prepared.bank.scenes if cluster else None
    xy = sf.per_image_bin_sums(prepared.x_hat, prepared.y_hat[:, 0], binning, setup)
    yy = sf.sample_pair_spectra(prepared.y_hat, binning, setup)
    cal = sf.calibration_ratios(xy, yy, min_shared=min_shared)

    intervals = {}
    for key in ("ratio_power", "ratio_cross_raw"):
        intervals[key] = sf.bootstrap_ci(
            lambda idx, k=key: sf.calibration_ratios(xy, yy, idx, min_shared)[k],
            n_images=xy.n_images, n_boot=n_boot, seed=seed + 6, groups=groups,
        )
    return {
        "binning": binning,
        "f_centers": binning.f_centers,
        **cal,
        "ratio_power_ci": intervals["ratio_power"],
        "ratio_cross_ci": intervals["ratio_cross_raw"],
    }


def run_oriented(prepared: Prepared) -> dict[str, Any]:
    """Compute coherence and fill-in resolved by orientation as well as radius.

    Args:
        prepared (Prepared): Spectra and binnings.

    Returns:
        dict[str, Any]: Grids shaped ``(n_radial, n_orient)`` for coherence, bits,
        and fill fraction, plus the bin centres needed to plot them.
    """
    binning, setup = prepared.oriented, prepared.setup
    spectra = sf.per_image_bin_sums(prepared.x_hat, prepared.y_hat[:, 0], binning, setup)
    reduced = sf.reduce_bin_sums(spectra)
    acc = sf.accumulate_fill(prepared.x_hat, prepared.y_hat, binning, setup)
    curves = sf.fill_curves(acc)
    return {
        "binning": binning,
        "f_centers": binning.f_centers,
        "theta_centers": binning.theta_centers,
        "gamma2": binning.reshape_oriented(reduced["gamma2"]),
        "bits": binning.reshape_oriented(reduced["bits"]),
        "fill_fraction": binning.reshape_oriented(curves["fill_fraction"]),
        "S_XX": binning.reshape_oriented(curves["S_XX"]),
        "S_YY": binning.reshape_oriented(curves["S_YY"]),
    }


def run_per_image(prepared: Prepared, cutoff: float) -> dict[str, Any]:
    """Score every target by how much of its reconstruction the prior supplies.

    Args:
        prepared (Prepared): Spectra and binnings.
        cutoff (float): Frequency splitting the low and high bands, in cycles/pixel.

    Returns:
        dict[str, Any]: Per-image fill scores, labels, and the sorted order.
    """
    scores = sf.per_image_fill_scores(prepared.x_hat, prepared.y_hat, prepared.setup, cutoff)
    order = np.argsort(scores["fill_overall"])
    return {
        **{key: value for key, value in scores.items()},
        "labels": prepared.bank.labels,
        "order": order,
        "cutoff": float(cutoff),
    }


def crossover_frequency(f_centers: np.ndarray, fill_fraction: np.ndarray, level: float = 0.5) -> float:
    """Find the frequency at which fill-in overtakes determination.

    Args:
        f_centers (numpy.ndarray): Radial bin centres, in cycles/pixel.
        fill_fraction (numpy.ndarray): Fill fraction per bin.
        level (float): Crossing level, ``0.5`` for the determined/filled crossover.

    Returns:
        float: Interpolated crossing frequency, or ``nan`` if the curve never crosses.
    """
    f = np.asarray(f_centers, dtype=float)
    values = np.asarray(fill_fraction, dtype=float)
    finite = np.isfinite(values)
    f, values = f[finite], values[finite]
    if values.min() > level or values.max() < level:
        return float("nan")
    above = np.flatnonzero(values >= level)
    if above.size == 0 or above[0] == 0:
        return float(f[0])
    i = above[0]
    lo, hi = values[i - 1], values[i]
    if hi == lo:
        return float(f[i])
    return float(f[i - 1] + (level - lo) / (hi - lo) * (f[i] - f[i - 1]))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _flatten(prefix: str, payload: dict[str, Any], out: dict[str, np.ndarray]) -> None:
    """Copy array-like entries of a result dict into a flat, savable mapping.

    Args:
        prefix (str): Key prefix identifying the analysis stage.
        payload (dict[str, Any]): Stage output, which may hold nested dicts.
        out (dict[str, numpy.ndarray]): Destination mapping, modified in place.
    """
    for key, value in payload.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            _flatten(name, value, out)
        elif isinstance(value, (np.ndarray, np.generic)):
            out[name] = np.asarray(value)
        elif isinstance(value, (int, float, bool)):
            out[name] = np.asarray(value)


def collect_arrays(**stages: dict[str, Any]) -> dict[str, np.ndarray]:
    """Flatten every analysis stage into one mapping suitable for ``np.savez``.

    Non-array entries such as binning objects and accumulations are dropped; the
    saved file is meant to hold results, not the machinery that produced them.

    Args:
        **stages (dict[str, Any]): Named stage outputs.

    Returns:
        dict[str, numpy.ndarray]: Flat mapping of ``stage/key`` to array.
    """
    out: dict[str, np.ndarray] = {}
    for prefix, payload in stages.items():
        _flatten(prefix, payload, out)
    return out


def build_summary(
    bank: Bank,
    prepared: Prepared,
    coherence: dict[str, Any],
    fill: dict[str, Any],
    calibration: dict[str, Any],
    per_image: dict[str, Any],
    nyquist: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Reduce every stage to the scalars and short lists the report quotes.

    Args:
        bank (Bank): The reconstruction bank.
        prepared (Prepared): Spectra and binnings.
        coherence (dict): Output of :func:`run_coherence`.
        fill (dict): Output of :func:`run_fill`.
        calibration (dict): Output of :func:`run_calibration`.
        per_image (dict): Output of :func:`run_per_image`.
        nyquist (dict): Output of :func:`nyquist_table`.

    Returns:
        dict[str, Any]: JSON-serializable summary.
    """
    n, k, height, width = bank.shape
    f = fill["f_centers"]
    crossover = crossover_frequency(f, fill["fill_fraction"])
    # The mosaic that samples most coarsely has the LOWEST Nyquist frequency, and
    # the one that samples most finely has the highest.  Naming these by the
    # mosaic rather than by the number avoids inverting the two.
    coarsest_mosaic = min(nyquist.items(), key=lambda kv: kv[1]["nyquist_min"])
    finest_mosaic = max(nyquist.items(), key=lambda kv: kv[1]["nyquist_min"])
    nyquist_low = coarsest_mosaic[1]["nyquist_min"]
    nyquist_high = finest_mosaic[1]["nyquist_min"]

    reliable = calibration["reliable"]
    power_ratio = calibration["ratio_power"]
    low = f <= nyquist_low
    high = f > nyquist_low

    summary = {
        "bank": {
            "path": str(bank.path),
            "n_images": n,
            "n_samples": k,
            "image_size": [height, width],
            "n_scenes": int(len(set(bank.scenes))),
            "scenes": sorted(set(bank.scenes.tolist())),
            "sampler": {
                key: bank.config.get(key)
                for key in ("model_id", "steps", "guidance_scale", "guidance_normalization",
                            "rate_noise_std", "cell_types", "base_seed")
            },
        },
        "mosaic_nyquist_cycles_per_pixel": nyquist,
        "nyquist_span": {
            "coarsest_mosaic": coarsest_mosaic[0],
            "coarsest_mosaic_nyquist": nyquist_low,
            "finest_mosaic": finest_mosaic[0],
            "finest_mosaic_nyquist": nyquist_high,
            "note": (
                "Each mosaic has a direction-dependent Nyquist; the inradius of its "
                "Brillouin zone is the conservative scalar. The mosaic that samples "
                "most coarsely has the lowest Nyquist and the one that samples most "
                "finely has the highest. The four mosaics are interleaved and jointly "
                "sample more densely than any one of them, so the finest single-mosaic "
                "Nyquist is a lower bound on the union's limit."
            ),
        },
        "information": {
            "single_sample_total_bits_per_image": coherence["single"]["total_bits"],
            "single_sample_total_bits_corrected": coherence["single"]["total_bits_corrected"],
            "single_sample_bits_per_pixel": coherence["single"]["total_bits"] / (height * width),
            "sample_mean_total_bits": coherence["mean"]["total_bits"],
            "sample_mean_debiased_total_bits": (
                coherence["mean_debiased"]["total_bits"] if coherence["mean_debiased"] else None
            ),
            "caveat": (
                "Gaussian-channel coherence lower bound on I(X;Y) (Borst & Theunissen, 1999). "
                "Non-Gaussian and non-linear dependence is invisible to it, so the true mutual "
                "information is at least this large and may be considerably larger."
            ),
        },
        "fill_in": {
            "crossover_frequency_cycles_per_pixel": crossover,
            "crossover_relative_to_coarsest_mosaic_nyquist": (
                float(crossover / nyquist_low) if np.isfinite(crossover) else None
            ),
            "crossover_relative_to_finest_mosaic_nyquist": (
                float(crossover / nyquist_high) if np.isfinite(crossover) else None
            ),
            "mean_fill_fraction_below_lowest_nyquist": float(np.nanmean(fill["fill_fraction"][low])),
            "mean_fill_fraction_above_lowest_nyquist": float(np.nanmean(fill["fill_fraction"][high])),
            "identity_residuals": {
                "law_of_total_variance_max_relative": fill["total_variance_max_relative_residual"],
                "error_decomposition_max_relative": fill["error_max_relative_residual"],
            },
            "error_share_calibrated_target": fill["calibrated_error_share"],
            "error_share_observed_median": float(np.nanmedian(fill["observed_error_share"])),
        },
        "calibration": {
            "power_ratio_median": float(np.nanmedian(power_ratio)),
            "power_ratio_below_lowest_nyquist": float(np.nanmedian(power_ratio[low])),
            "power_ratio_above_lowest_nyquist": float(np.nanmedian(power_ratio[high])),
            "log2_power_ratio_median": float(np.nanmedian(calibration["log2_ratio_power"])),
            "cross_ratio_median_where_reliable": (
                float(np.nanmedian(calibration["ratio_cross"][reliable])) if reliable.any() else None
            ),
            "n_reliable_bins": int(reliable.sum()),
            "n_bins": int(len(f)),
            "interpretation": (
                "S_YY/S_XX below one means the sampler under-expresses power at that "
                "frequency; above one means it injects more than the prior would. "
                "|S_XY|/S_YjYk below one means two samples agree with each other more "
                "than either agrees with the truth, the signature of a shared bias. "
                "The power ratio is blind to a wrong posterior mean and the cross ratio "
                "is blind to a wrong posterior spread, so both must be read."
            ),
        },
        "per_image": {
            "cutoff_cycles_per_pixel": per_image["cutoff"],
            "fill_overall_mean": float(np.mean(per_image["fill_overall"])),
            "fill_overall_std": float(np.std(per_image["fill_overall"])),
            "fill_below_mean": float(np.mean(per_image["fill_below"])),
            "fill_above_mean": float(np.mean(per_image["fill_above"])),
            "most_determined": [
                {"label": str(per_image["labels"][i]), "fill_overall": float(per_image["fill_overall"][i])}
                for i in per_image["order"][:5]
            ],
            "most_filled_in": [
                {"label": str(per_image["labels"][i]), "fill_overall": float(per_image["fill_overall"][i])}
                for i in per_image["order"][-5:][::-1]
            ],
        },
        "targets": {
            "mode": bank.config.get("target_mode", "tile"),
            "bandwidth": bank.config.get("bandwidth"),
            "content_limit_cycles_per_pixel": (
                (bank.config.get("bandwidth") or {}).get("content_limit_min", 0.5)
            ),
            "note": (
                "Above the content limit S_XX reflects resampling rather than scene "
                "structure, so the power-ratio diagnostic is not interpretable there. "
                "The fill fraction is unaffected, since it is computed from Y alone."
            ),
        },
        "preprocessing": {
            "window": "hann/tukey as configured; see analysis notebook",
            "window_power_normalization_U": float(prepared.setup.window_power),
            "dc_excluded": True,
            "real_degrees_of_freedom_per_image": int(height * width),
            "independent_complex_modes_per_image": float(
                sf.independent_complex_modes((height, width))
            ),
        },
    }
    return summary
