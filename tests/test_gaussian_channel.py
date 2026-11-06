"""Tests 2 and 3: the additive Gaussian channel and the independence null.

Test 2 is the central correctness check on the estimator.  X is a Gaussian random
field with a ``1/f^2`` power spectrum and ``Y = X + noise`` with a known noise
spectrum, so the coherence and the information rate both have exact closed forms
and the estimator must reproduce them across the whole frequency range.

Test 3 checks the other end: when Y carries nothing about X, the shuffle-corrected
coherence and the information must be consistent with zero rather than with the
positive bias that a raw finite-sample coherence always shows.
"""

from __future__ import annotations

import numpy as np
import pytest

import analysis.spectral_fillin as sf
import toy

SHAPE = (64, 64)
N_IMAGES = 400
N_BINS = 16


@pytest.fixture(scope="module")
def channel():
    """Build a 1/f^2 field observed through a known additive-noise channel.

    Returns:
        dict: Spectra, setup, binning, and the analytic per-bin predictions.
    """
    rng = np.random.default_rng(20240607)
    prior = toy.power_law_spectrum(SHAPE, exponent=2.0)
    # A flat noise floor set near the median signal power puts the coherence
    # crossover in the middle of the band, where the estimator is best resolved
    # and where a scale error would be most visible.
    noise = np.full(SHAPE, float(np.median(prior)) * 3.0)

    x = toy.gaussian_field(prior, N_IMAGES, rng)
    y = x + toy.gaussian_field(noise, N_IMAGES, rng)

    setup = sf.make_setup(SHAPE, "none")
    binning = sf.radial_binning(setup, N_BINS, f_max=0.5)
    x_hat, y_hat = sf.preprocess(x, setup), sf.preprocess(y, setup)

    # The bin-averaged coherence of this channel is exact, not approximate:
    # S_XY = Pbar, S_XX = Pbar and S_YY = Pbar + Nbar within a bin, so
    # gamma^2 = Pbar / (Pbar + Nbar) with no Jensen correction needed.
    p_bar = sf.bin_mean_of_modes(prior, binning, setup)
    n_bar = sf.bin_mean_of_modes(noise, binning, setup)
    snr = p_bar / n_bar
    return {
        "setup": setup,
        "binning": binning,
        "x_hat": x_hat,
        "y_hat": y_hat,
        "spectra": sf.per_image_bin_sums(x_hat, y_hat, binning, setup),
        "snr": snr,
        "gamma2_expected": snr / (1.0 + snr),
        "bits_per_dof_expected": 0.5 * np.log2(1.0 + snr),
        "prior": prior,
        "noise": noise,
    }


def test_marginal_spectra_match_the_generator(channel):
    """The estimated auto-spectra reproduce the spectra the fields were drawn with."""
    reduced = sf.reduce_bin_sums(channel["spectra"])
    p_bar = sf.bin_mean_of_modes(channel["prior"], channel["binning"], channel["setup"])
    n_bar = sf.bin_mean_of_modes(channel["noise"], channel["binning"], channel["setup"])
    np.testing.assert_allclose(reduced["S_XX"], p_bar, rtol=0.05)
    np.testing.assert_allclose(reduced["S_YY"], p_bar + n_bar, rtol=0.05)


def test_coherence_matches_analytic_value(channel):
    """Estimated coherence equals SNR/(1+SNR) in every bin, within 2 percent."""
    gamma2 = sf.reduce_bin_sums(channel["spectra"])["gamma2"]
    np.testing.assert_allclose(gamma2, channel["gamma2_expected"], rtol=0.02)


def test_information_matches_analytic_value(channel):
    """Estimated information equals 0.5*log2(1+SNR) bits per real degree of freedom."""
    bits = sf.reduce_bin_sums(channel["spectra"])["bits_per_dof"]
    np.testing.assert_allclose(bits, channel["bits_per_dof_expected"], rtol=0.04)


def test_analytic_value_lies_inside_the_bootstrap_interval(channel):
    """The 95 percent bootstrap interval covers the truth in essentially every bin."""
    spectra = channel["spectra"]
    ci = sf.bootstrap_ci(
        lambda idx: sf.reduce_bin_sums(spectra, idx)["gamma2"],
        n_images=spectra.n_images,
        n_boot=300,
        seed=3,
    )
    expected = channel["gamma2_expected"]
    covered = (ci["lo"] <= expected) & (expected <= ci["hi"])
    # A 95 percent interval is expected to miss about one bin in twenty by
    # construction, so the suite asserts coverage across bins rather than
    # demanding every single one.
    assert covered.mean() >= 0.85, f"coverage {covered.mean():.2f}, missed bins {np.flatnonzero(~covered)}"


def test_coherence_is_not_an_average_of_per_mode_coherences(channel):
    """Averaging per-mode coherences inflates the answer; the estimator must not do it."""
    x_hat, y_hat = channel["x_hat"], channel["y_hat"]
    binning, setup = channel["binning"], channel["setup"]
    per_mode = np.abs((x_hat * np.conj(y_hat)).mean(axis=0)) ** 2 / (
        (np.abs(x_hat) ** 2).mean(axis=0) * (np.abs(y_hat) ** 2).mean(axis=0)
    )
    wrong = sf.bin_mean_of_modes(per_mode, binning, setup)
    right = sf.reduce_bin_sums(channel["spectra"])["gamma2"]
    high = channel["snr"] < 0.5
    assert np.all(wrong[high] > right[high]), "per-mode averaging should be biased upward"
    np.testing.assert_allclose(right, channel["gamma2_expected"], rtol=0.02)


def test_shuffle_null_is_small_against_a_real_signal(channel):
    """The null floor is far below the true coherence wherever there is signal."""
    null = sf.shuffle_null_coherence(
        channel["x_hat"], channel["y_hat"], channel["binning"], channel["setup"], n_shuffles=6, seed=11
    )
    gamma2 = sf.reduce_bin_sums(channel["spectra"])["gamma2"]
    assert np.all(null["gamma2_null"] < 0.02)
    assert np.all(null["gamma2_null"] < 0.25 * gamma2)


@pytest.fixture(scope="module")
def null_pair():
    """Build a target and an independent reconstruction with matched marginal spectra.

    Returns:
        dict: Spectra, setup, binning, and the raw and corrected coherence curves.
    """
    rng = np.random.default_rng(99)
    prior = toy.power_law_spectrum(SHAPE, exponent=2.0)
    x = toy.gaussian_field(prior, N_IMAGES, rng)
    y = toy.gaussian_field(prior, N_IMAGES, rng)

    setup = sf.make_setup(SHAPE, "none")
    binning = sf.radial_binning(setup, N_BINS, f_max=0.5)
    x_hat, y_hat = sf.preprocess(x, setup), sf.preprocess(y, setup)
    spectra = sf.per_image_bin_sums(x_hat, y_hat, binning, setup)
    null = sf.shuffle_null_coherence(x_hat, y_hat, binning, setup, n_shuffles=8, seed=5)
    raw = sf.reduce_bin_sums(spectra)["gamma2"]
    return {
        "setup": setup,
        "binning": binning,
        "spectra": spectra,
        "raw": raw,
        "null": null,
        "corrected": sf.correct_coherence(raw, null["gamma2_null"]),
    }


def test_null_raw_coherence_is_small_but_positive(null_pair):
    """Independent fields still show a positive finite-sample coherence."""
    raw = null_pair["raw"]
    assert np.all(raw < 0.02)
    assert np.all(raw > 0.0), "raw coherence is biased strictly upward, never exactly zero"


def test_null_corrected_coherence_is_consistent_with_zero(null_pair):
    """Shuffle correction removes the floor, leaving coherence at zero."""
    corrected = null_pair["corrected"]
    assert np.all(corrected < 5e-3)
    assert corrected.mean() < 1e-3


def test_null_information_is_consistent_with_zero(null_pair):
    """The corrected information is a vanishing fraction of the channel's capacity."""
    binning = null_pair["binning"]
    bits = sf.information_bits(null_pair["corrected"], binning.dof).sum()
    per_pixel = bits / (SHAPE[0] * SHAPE[1])
    assert per_pixel < 0.01, f"null leaks {per_pixel:.4f} bits/pixel"


def test_null_bootstrap_interval_includes_zero(null_pair):
    """The bootstrap interval on corrected coherence covers zero in most bins."""
    spectra, null = null_pair["spectra"], null_pair["null"]
    ci = sf.bootstrap_ci(
        lambda idx: sf.correct_coherence(sf.reduce_bin_sums(spectra, idx)["gamma2"], null["gamma2_null"]),
        n_images=spectra.n_images,
        n_boot=300,
        seed=7,
    )
    assert (ci["lo"] <= 0.0).mean() >= 0.85
