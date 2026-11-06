"""Test 4: a linear-Gaussian toy RGC whose posterior can be sampled exactly.

``Z = blur(X) + noise`` with a known convolution makes the posterior diagonal in
the Fourier domain, so reconstructions can be drawn from the true posterior with
no sampler approximation at all.  Everything the fill-in analysis claims then has
an exact target: the fill fraction must be ``1 / (1 + SNR(f))``, the
exchangeability diagnostics must pass because the sampler really is exact, and
the two variance identities must close.

A calibrated sampler is also the only setting in which a failure is unambiguous.
On the real model a diagnostic that deviates could mean either a miscalibrated
sampler or a broken estimator; here it can only mean the latter.
"""

from __future__ import annotations

import numpy as np
import pytest

import analysis.spectral_fillin as sf
import toy

SHAPE = (64, 64)
N_IMAGES = 200
N_SAMPLES = 16
N_BINS = 16


@pytest.fixture(scope="module")
def exact():
    """Draw targets and exact posterior reconstructions from the diagonal toy.

    Returns:
        dict: Model, spectra, binning, and the pre-built accumulations.
    """
    rng = np.random.default_rng(424242)
    model = toy.make_diagonal_posterior(SHAPE, prior_exponent=2.0, blur_sigma=3.0)
    x = model.draw_targets(N_IMAGES, rng)
    z = model.encode(x, rng)
    y = model.sample(z, N_SAMPLES, rng)

    setup = sf.make_setup(SHAPE, "none")
    binning = sf.radial_binning(setup, N_BINS, f_max=0.5)
    x_hat = sf.preprocess(x, setup)
    y_hat = sf.preprocess(y.reshape(-1, *SHAPE), setup).reshape(N_IMAGES, N_SAMPLES, *SHAPE)
    return {
        "model": model,
        "setup": setup,
        "binning": binning,
        "x_hat": x_hat,
        "y_hat": y_hat,
        "acc": sf.accumulate_fill(x_hat, y_hat, binning, setup),
        "xy": sf.per_image_bin_sums(x_hat, y_hat[:, 0], binning, setup),
        "yy": sf.sample_pair_spectra(y_hat, binning, setup),
    }


def test_fill_fraction_matches_inverse_snr_per_mode(exact):
    """Per mode, the fill fraction is 1/(1+SNR(f)) as the exact posterior requires."""
    model, setup = exact["model"], exact["setup"]
    y_hat = exact["y_hat"]
    v_fill = sf.fill_variance_per_mode(y_hat).mean(axis=0)
    s_yy = (np.abs(y_hat) ** 2).mean(axis=(0, 1))
    estimated = v_fill / s_yy
    expected = model.expected_fill_fraction()

    valid = ~setup.dc_mask
    error = np.abs(estimated[valid] - expected[valid])
    assert np.median(error) < 0.02
    assert np.mean(error) < 0.03
    assert np.percentile(error, 95) < 0.06


def test_fill_fraction_matches_prediction_per_bin(exact):
    """Binned, the fill fraction matches binned V/P, not 1/(1+binned SNR)."""
    model, setup, binning = exact["model"], exact["setup"], exact["binning"]
    curves = sf.fill_curves(exact["acc"])
    v_bar = sf.bin_mean_of_modes(model.posterior_variance, binning, setup)
    p_bar = sf.bin_mean_of_modes(model.prior_spectrum, binning, setup)
    np.testing.assert_allclose(curves["fill_fraction"], v_bar / p_bar, atol=0.02)

    # The naive binned predictor is a different number wherever SNR varies
    # sharply inside a bin; binning does not commute with 1/(1+SNR).  This is
    # asserted so the distinction is not quietly lost later.
    snr_bar = sf.bin_mean_of_modes(model.snr, binning, setup)
    naive = 1.0 / (1.0 + snr_bar)
    assert np.max(np.abs(naive - v_bar / p_bar)) > 0.05


def test_determined_and_fill_fractions_sum_to_one(exact):
    """The determined fraction is exactly the complement of the fill fraction."""
    curves = sf.fill_curves(exact["acc"])
    np.testing.assert_allclose(
        curves["fill_fraction"] + curves["determined_fraction"], 1.0, rtol=1e-12
    )


def test_fill_fraction_is_within_bootstrap_interval(exact):
    """The exact prediction lies inside the bootstrap interval in most bins."""
    acc, model = exact["acc"], exact["model"]
    binning, setup = exact["binning"], exact["setup"]
    ci = sf.bootstrap_ci(
        lambda idx: sf.fill_curves(acc, idx)["fill_fraction"],
        n_images=acc.n_images,
        n_boot=300,
        seed=13,
    )
    expected = sf.bin_mean_of_modes(model.posterior_variance, binning, setup) / sf.bin_mean_of_modes(
        model.prior_spectrum, binning, setup
    )
    covered = (ci["lo"] <= expected) & (expected <= ci["hi"])
    assert covered.mean() >= 0.80, f"coverage {covered.mean():.2f}"


def test_exchangeability_power_ratio_is_one(exact):
    """An exact posterior sampler reproduces the prior, so S_YY equals S_XX."""
    cal = sf.calibration_ratios(exact["xy"], exact["yy"])
    np.testing.assert_allclose(cal["ratio_power"], 1.0, atol=0.06)
    assert abs(np.median(cal["log2_ratio_power"])) < 0.03


def test_exchangeability_cross_ratio_is_one_where_measurable(exact):
    """A target shares as much with a sample as two samples share, where anything is shared."""
    cal = sf.calibration_ratios(exact["xy"], exact["yy"], min_shared=0.05)
    reliable = cal["reliable"]
    assert reliable.sum() >= 4, "the toy must leave some bins with shared structure"
    np.testing.assert_allclose(cal["ratio_cross"][reliable], 1.0, atol=0.08)


def test_cross_ratio_is_masked_where_nothing_is_shared(exact):
    """Bins with no determined structure are flagged, not reported as miscalibration."""
    cal = sf.calibration_ratios(exact["xy"], exact["yy"], min_shared=0.05)
    assert np.any(~cal["reliable"]), "the toy must saturate at high frequency"
    assert np.all(np.isnan(cal["ratio_cross"][~cal["reliable"]]))
    # Without the mask those bins are wild, which is exactly why the mask exists.
    assert np.nanmax(cal["ratio_cross_raw"][~cal["reliable"]]) > 2.0


def test_law_of_total_variance_holds(exact):
    """S_YY equals the explained plus unexplained parts, to numerical precision."""
    check = sf.total_variance_check(exact["acc"])
    np.testing.assert_allclose(check["lhs"], check["rhs"], rtol=1e-10)
    assert np.max(np.abs(check["relative_residual"])) < 1e-10


def test_law_of_total_variance_needs_the_finite_k_correction(exact):
    """Skipping the V/K debias breaks the identity by a visible amount."""
    check = sf.total_variance_check(exact["acc"])
    naive_rhs = check["explained_raw"] + check["unexplained"]
    assert np.max(np.abs(naive_rhs - check["lhs"]) / check["lhs"]) > 1e-3


def test_error_decomposition_holds(exact):
    """The per-bin squared error splits exactly into a bias and a fill-in term.

    The residual is compared against the power in the bin rather than against the
    error itself.  With a ``1 / f^2`` prior the spectrum spans four orders of
    magnitude, and at low frequency the reconstruction error is a tiny difference
    between two large numbers, so a residual that is negligible next to the
    bin's power is still a part in ten thousand of the error.  That is float64
    cancellation in the test's own inputs, not slack in the identity.
    """
    decomposition = sf.error_decomposition(exact["acc"])
    power = sf.fill_curves(exact["acc"])["S_YY"]
    assert np.max(np.abs(decomposition["residual"]) / power) < 1e-12
    assert np.all(decomposition["bias_term"] > 0)
    assert np.all(decomposition["fill_term"] > 0)


def test_calibrated_sampler_splits_error_in_a_fixed_ratio(exact):
    """For an exact posterior sampler the fill-in share of the error is (K-1)/(2K).

    Exchangeability of X and Y given Z forces ``E|Xhat - Yhat|^2 = 2 Var(Yhat|Z)``
    at **every** frequency, because the target is itself just another draw from
    the posterior.  The fill-in term is ``((K-1)/K) Var``, so its share of the
    total error is the frequency-independent constant ``(K-1)/(2K)``.  A real
    sampler that is too smooth will fall below this line at high frequency, which
    makes the ratio a calibration diagnostic in its own right.
    """
    decomposition = sf.error_decomposition(exact["acc"])
    share = decomposition["fill_term"] / decomposition["lhs"]
    expected = (N_SAMPLES - 1) / (2.0 * N_SAMPLES)
    np.testing.assert_allclose(share, expected, atol=0.03)


def test_sample_pair_closed_form_matches_brute_force(exact):
    """S_{Y_j Y_k} = S_YY - V_fill is checked against an explicit loop over pairs."""
    y_hat = exact["y_hat"][:20, :6]
    n, k = y_hat.shape[:2]
    brute = np.zeros(y_hat.shape[-2:], dtype=np.complex128)
    count = 0
    for j in range(k):
        for m in range(k):
            if j == m:
                continue
            brute += (y_hat[:, j] * np.conj(y_hat[:, m])).mean(axis=0)
            count += 1
    brute /= count
    closed = (np.abs(y_hat) ** 2).mean(axis=1).mean(axis=0) - sf.fill_variance_per_mode(y_hat).mean(axis=0)
    np.testing.assert_allclose(brute.real, closed, rtol=1e-10)
    assert np.max(np.abs(brute.imag)) < 1e-10 * np.max(np.abs(brute.real))


def test_sample_coherence_equals_one_minus_fill_squared(exact):
    """Sample-versus-sample coherence is the square of the determined fraction."""
    gamma2_jk = sf.reduce_bin_sums(exact["yy"])["gamma2"]
    determined = sf.fill_curves(exact["acc"])["determined_fraction"]
    np.testing.assert_allclose(gamma2_jk, np.clip(determined, 0, None) ** 2, atol=1e-6)


def test_single_sample_coherence_is_below_sample_mean_coherence(exact):
    """Averaging samples removes fill-in noise, so the sample mean tracks X better."""
    single = sf.reduce_bin_sums(exact["xy"])["gamma2"]
    split = sf.split_half_mean_spectra(exact["x_hat"], exact["y_hat"], exact["binning"], exact["setup"], seed=1)
    mean_based = sf.debiased_mean_coherence(split)["gamma2_debiased"]
    informative = single > 0.02
    assert np.all(mean_based[informative] >= single[informative] - 1e-6)
