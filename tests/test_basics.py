"""Test 1: Parseval, Hermitian mode counting, and DC exclusion.

These are the bookkeeping guarantees every later test rests on.  If the window
normalization or the degree-of-freedom count is wrong, every spectrum and every
bit count downstream is wrong by a constant factor that no other test would
isolate.
"""

from __future__ import annotations

import numpy as np
import pytest

import diffusion.analysis.spectral_fillin as sf

SHAPES = [(64, 64), (63, 64), (64, 63), (63, 65), (32, 48)]
WINDOWS = [("none", 0.0), ("hann", 0.0), ("tukey", 0.25), ("tukey", 0.5), ("tukey", 1.0)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("window,alpha", WINDOWS)
def test_parseval_holds_with_window_normalization(shape, window, alpha):
    """Total spectral power equals total windowed image power divided by U."""
    setup = sf.make_setup(shape, window, alpha)
    rng = np.random.default_rng(0)
    images = rng.normal(size=(3,) + shape)

    x_hat = sf.preprocess(images, setup)
    spectral_power = (np.abs(x_hat) ** 2).sum(axis=(-2, -1))

    centred = images - images.mean(axis=(-2, -1), keepdims=True)
    image_power = ((centred * setup.window) ** 2).sum(axis=(-2, -1)) / setup.window_power

    np.testing.assert_allclose(spectral_power, image_power, rtol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_window_normalization_keeps_total_power_comparable(shape):
    """Windowed and unwindowed spectra agree on total power to within sampling noise.

    The whole point of dividing by ``U = mean(window**2)`` is that the window
    should taper the edges without changing the power budget, so that spectra
    computed with different windows remain comparable.
    """
    rng = np.random.default_rng(1)
    images = rng.normal(size=(64,) + shape)
    totals = {}
    for window, alpha in [("none", 0.0), ("hann", 0.0), ("tukey", 0.5)]:
        setup = sf.make_setup(shape, window, alpha)
        totals[window] = (np.abs(sf.preprocess(images, setup)) ** 2).sum(axis=(-2, -1)).mean()
    assert totals["hann"] == pytest.approx(totals["none"], rel=0.05)
    assert totals["tukey"] == pytest.approx(totals["none"], rel=0.05)


@pytest.mark.parametrize("shape", SHAPES)
def test_real_degrees_of_freedom_total_is_h_times_w(shape):
    """The full plane carries exactly H*W real degrees of freedom, even or odd."""
    setup = sf.make_setup(shape)
    assert setup.dof_weight.sum() == pytest.approx(shape[0] * shape[1])
    assert setup.dof_weight.shape == shape


@pytest.mark.parametrize("shape", SHAPES)
def test_independent_complex_mode_count(shape):
    """A real image has H*W/2 independent complex modes, one half-plane's worth."""
    total = sf.independent_complex_modes(shape, exclude_dc=False)
    assert total == pytest.approx(shape[0] * shape[1] / 2.0)
    assert sf.independent_complex_modes(shape, exclude_dc=True) == pytest.approx(total - 0.5)


@pytest.mark.parametrize("shape", SHAPES)
def test_binning_accounts_for_every_mode_once(shape):
    """With f_max covering the whole plane, bins hold every real degree of freedom but DC."""
    setup = sf.make_setup(shape)
    binning = sf.radial_binning(setup, n_bins=24, f_max=0.71)
    assert binning.dof.sum() == pytest.approx(shape[0] * shape[1] - 1)
    assert binning.n_excluded == 1
    assert binning.n_modes.sum() == shape[0] * shape[1] - 1


@pytest.mark.parametrize("shape", SHAPES)
def test_oriented_binning_accounts_for_every_mode_once(shape):
    """The radius-by-orientation grid partitions the plane the same way."""
    setup = sf.make_setup(shape)
    binning = sf.radial_orientation_binning(setup, n_bins=12, n_orient=8, f_max=0.71)
    assert binning.dof.sum() == pytest.approx(shape[0] * shape[1] - 1)
    assert binning.is_oriented
    assert binning.reshape_oriented(binning.dof).shape == (12, 8)


@pytest.mark.parametrize("shape", SHAPES)
def test_dc_is_excluded_and_mean_offsets_are_invisible(shape):
    """DC is never binned, and adding a constant to an image changes nothing."""
    setup = sf.make_setup(shape)
    binning = sf.radial_binning(setup, n_bins=16)
    assert binning.index[0, 0] == -1

    rng = np.random.default_rng(2)
    images = rng.normal(size=(4,) + shape)
    offsets = rng.normal(size=(4, 1, 1)) * 10.0
    base = sf.per_image_bin_sums(sf.preprocess(images, setup), sf.preprocess(images, setup), binning, setup)
    shifted = sf.preprocess(images + offsets, setup)
    moved = sf.per_image_bin_sums(shifted, shifted, binning, setup)
    np.testing.assert_allclose(base.s_xx, moved.s_xx, rtol=1e-10)


def test_information_uses_real_degrees_of_freedom():
    """Half-log per real degree of freedom equals full-log per independent complex mode."""
    gamma2 = np.array([0.0, 0.25, 0.5, 0.9])
    shape = (64, 64)
    real_dof = np.full(4, shape[0] * shape[1], dtype=float)
    complex_modes = real_dof / 2.0
    half_log_over_real = sf.information_bits(gamma2, real_dof)
    full_log_over_complex = -np.log2(1.0 - gamma2) * complex_modes
    np.testing.assert_allclose(half_log_over_real, full_log_over_complex, rtol=1e-12)


def test_information_matches_gaussian_channel_capacity():
    """-0.5*log2(1-gamma^2) is 0.5*log2(1+SNR) per real degree of freedom."""
    snr = np.array([0.1, 1.0, 10.0, 100.0])
    gamma2 = snr / (1.0 + snr)
    np.testing.assert_allclose(
        sf.information_bits(gamma2, np.ones_like(snr)), 0.5 * np.log2(1.0 + snr), rtol=1e-12
    )


def test_hex_nyquist_geometry():
    """The triangular-lattice Brillouin zone has the right radii and corner directions."""
    spacing = 10.0
    inradius, circumradius = sf.hex_nyquist_bounds(spacing)
    assert inradius == pytest.approx(1.0 / (np.sqrt(3.0) * spacing))
    assert circumradius == pytest.approx(2.0 / (3.0 * spacing))
    assert circumradius / inradius == pytest.approx(2.0 / np.sqrt(3.0))

    theta = np.linspace(0, 2 * np.pi, 721)
    curve = sf.hex_nyquist(spacing, theta)
    assert curve.min() == pytest.approx(inradius, rel=1e-6)
    assert curve.max() == pytest.approx(circumradius, rel=1e-6)
    # Corners every 60 degrees starting at 0; face normals halfway between.
    np.testing.assert_allclose(sf.hex_nyquist(spacing, np.deg2rad([0, 60, 120, 180])), circumradius, rtol=1e-9)
    np.testing.assert_allclose(sf.hex_nyquist(spacing, np.deg2rad([30, 90, 150])), inradius, rtol=1e-9)
    # Six-fold symmetry.
    np.testing.assert_allclose(curve, sf.hex_nyquist(spacing, theta + np.pi / 3.0), rtol=1e-9)


def test_square_nyquist():
    """A square lattice of spacing S has Nyquist 1/(2S)."""
    assert sf.square_nyquist(4.0) == pytest.approx(0.125)
    with pytest.raises(ValueError):
        sf.square_nyquist(0.0)
