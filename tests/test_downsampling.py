"""Downsampling toy: a qualitative check that the Nyquist overlay lands correctly.

Subsampling is not a convolution, so aliasing couples each frequency to its
images under the sampling lattice and the posterior stops being diagonal in the
Fourier domain.  There is therefore no analytic fill fraction to compare against,
which is why this is a sanity check on the overlay rather than an accuracy test.
The posterior is still exactly Gaussian, so at 32x32 it is formed densely and
sampled without approximation; what is being checked is the *shape* of the curve
against the lattice Nyquist, not a number.

The expectation is simple and strong: below ``1 / (2 * spacing)`` the lattice
measures the image and the reconstruction is determined; above it the lattice
cannot distinguish frequencies and the prior must invent them, so the fill
fraction rises to one.
"""

from __future__ import annotations

import numpy as np
import pytest

import diffusion.analysis.spectral_fillin as sf
import toy
from conftest import ARTIFACT_DIR

SHAPE = (32, 32)
SPACING = 4
N_IMAGES = 160
N_SAMPLES = 8
N_BINS = 12


@pytest.fixture(scope="module")
def lattice():
    """Run the blur-then-subsample toy with its exact dense posterior.

    Returns:
        dict: Fill curves, binning, the Nyquist frequency, and the raw arrays.
    """
    rng = np.random.default_rng(31337)
    model = toy.make_downsampling_posterior(SHAPE, spacing=SPACING, blur_sigma=1.5)
    x = model.draw_targets(N_IMAGES, rng)
    z = model.encode(x, rng)
    y = model.sample(z, N_SAMPLES, rng)

    setup = sf.make_setup(SHAPE, "none")
    binning = sf.radial_binning(setup, N_BINS, f_max=0.5)
    x_hat = sf.preprocess(x, setup)
    y_hat = sf.preprocess(y.reshape(-1, *SHAPE), setup).reshape(N_IMAGES, N_SAMPLES, *SHAPE)
    acc = sf.accumulate_fill(x_hat, y_hat, binning, setup)
    return {
        "model": model,
        "setup": setup,
        "binning": binning,
        "x": x,
        "y": y,
        "x_hat": x_hat,
        "y_hat": y_hat,
        "acc": acc,
        "curves": sf.fill_curves(acc),
        "nyquist": sf.square_nyquist(SPACING),
    }


def test_fill_fraction_rises_past_the_lattice_nyquist(lattice):
    """Fill-in is small below the lattice Nyquist and near total above it."""
    f = lattice["binning"].f_centers
    fill = lattice["curves"]["fill_fraction"]
    nyquist = lattice["nyquist"]
    below = fill[f < 0.7 * nyquist]
    above = fill[f > 1.5 * nyquist]
    assert below.max() < 0.3, f"too much fill-in below Nyquist: {below.max():.3f}"
    assert above.min() > 0.85, f"too little fill-in above Nyquist: {above.min():.3f}"


def test_crossover_is_located_at_the_lattice_nyquist(lattice):
    """The half-filled frequency sits within a bin width of 1/(2S)."""
    f = lattice["binning"].f_centers
    fill = lattice["curves"]["fill_fraction"]
    crossover = np.interp(0.5, fill, f)
    bin_width = f[1] - f[0]
    assert abs(crossover - lattice["nyquist"]) < bin_width, (
        f"crossover at {crossover:.4f} but lattice Nyquist is {lattice['nyquist']:.4f}"
    )


def test_fill_fraction_is_monotonic_in_frequency(lattice):
    """A lowpass-then-subsample forward model gives a monotonically rising curve."""
    fill = lattice["curves"]["fill_fraction"]
    assert np.all(np.diff(fill) > -0.02)


def test_identities_still_hold_without_a_diagonal_posterior(lattice):
    """The variance identities are algebraic, so aliasing does not threaten them."""
    power = lattice["curves"]["S_YY"]
    total = sf.total_variance_check(lattice["acc"])
    decomposition = sf.error_decomposition(lattice["acc"])
    assert np.max(np.abs(total["residual"]) / power) < 1e-12
    assert np.max(np.abs(decomposition["residual"]) / power) < 1e-12


def test_exact_sampler_still_passes_exchangeability(lattice):
    """An exact posterior sampler is exchangeable whether or not the posterior is diagonal."""
    xy = sf.per_image_bin_sums(lattice["x_hat"], lattice["y_hat"][:, 0], lattice["binning"], lattice["setup"])
    yy = sf.sample_pair_spectra(lattice["y_hat"], lattice["binning"], lattice["setup"])
    cal = sf.calibration_ratios(xy, yy, min_shared=0.05)
    np.testing.assert_allclose(cal["ratio_power"], 1.0, atol=0.1)
    np.testing.assert_allclose(cal["ratio_cross"][cal["reliable"]], 1.0, atol=0.12)


def test_per_image_scores_split_at_the_nyquist(lattice):
    """Per-image scores show far more fill-in above the lattice Nyquist than below."""
    scores = sf.per_image_fill_scores(
        lattice["x_hat"], lattice["y_hat"], lattice["setup"], cutoff=lattice["nyquist"]
    )
    assert scores["fill_above"].mean() > scores["fill_below"].mean() + 0.3
    assert np.all((scores["fill_overall"] >= 0) & (scores["fill_overall"] <= 1.05))
    assert scores["fill_overall"].shape == (N_IMAGES,)
    # The scores must actually separate images, otherwise sorting by them is useless.
    assert scores["fill_overall"].std() > 1e-3


def test_pixel_variance_maps_are_consistent_with_spectral_fill(lattice):
    """Pixel-space within-Z variance totals agree with the spectral fill-in variance."""
    maps = sf.pixel_variance_maps(lattice["y"])
    assert maps["mean"].shape == SHAPE
    assert maps["per_image"].shape == (N_IMAGES,) + SHAPE
    assert maps["sample_mean"].shape == (N_IMAGES,) + SHAPE
    assert maps["per_image_scalar"].shape == (N_IMAGES,)
    spectral_total = sf.fill_variance_per_mode(lattice["y_hat"]).sum(axis=(-2, -1)).mean()
    # Parseval with no window: summed spectral variance equals H*W times the mean
    # pixel variance, up to the mean removal that preprocess applies.
    pixel_total = maps["per_image"].mean() * SHAPE[0] * SHAPE[1]
    assert spectral_total == pytest.approx(pixel_total, rel=0.02)


def test_overlay_figure_is_written(lattice):
    """Save the qualitative overlay figure that this toy exists to produce."""
    plt = pytest.importorskip("matplotlib.pyplot")
    f = lattice["binning"].f_centers
    acc = lattice["acc"]
    ci = sf.bootstrap_ci(
        lambda idx: sf.fill_curves(acc, idx)["fill_fraction"], n_images=N_IMAGES, n_boot=200, seed=8
    )

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.fill_between(f, ci["lo"], ci["hi"], alpha=0.25, color="C0", label="95% CI")
    axis.plot(f, ci["point"], "o-", color="C0", label="fill fraction $F(b)$")
    axis.plot(f, 1.0 - ci["point"], "s--", color="C1", label="determined $1-F$")
    axis.axvline(lattice["nyquist"], color="k", ls=":", lw=1.5,
                 label=f"square lattice Nyquist $1/(2S)$, S={SPACING}")
    axis.set_xlabel("spatial frequency (cycles/pixel)")
    axis.set_ylabel("fraction of reconstruction power")
    axis.set_title("Blur-then-subsample toy: fill-in rises past the lattice Nyquist")
    axis.set_ylim(-0.05, 1.05)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "downsampling_nyquist_overlay.png", dpi=150)
    plt.close(figure)
    assert (ARTIFACT_DIR / "downsampling_nyquist_overlay.png").exists()
