"""Test 5: the calibration diagnostics must detect a miscalibrated sampler.

The same exact toy as test 4, deliberately broken in three ways, to establish
that each diagnostic fires in the right direction and, just as importantly, to
establish which miscalibration each diagnostic is blind to.

The two failure modes are not interchangeable:

* **Posterior noise scaled by c.**  The sampler draws the right mean but with
  the wrong spread.  ``S_YY = P + (c^2 - 1) V``, so the power ratio moves below
  one for ``c < 1`` and above one for ``c > 1``.  The cross ratio does not move
  at all, because scaling the noise leaves both ``S_XY`` and ``S_{Y_j Y_k}``
  equal to ``P - V``.
* **Posterior mean scaled by g**, the stand-in for wrong guidance strength.  Then
  ``S_XY = g (P - V)`` while ``S_{Y_j Y_k} = g^2 (P - V)``, so the cross ratio is
  exactly ``1 / g`` and the power ratio moves too.

A guided diffusion sampler can be wrong in both ways at once, so the report must
read both curves rather than either alone.
"""

from __future__ import annotations

import numpy as np
import pytest

import analysis.spectral_fillin as sf
import toy

SHAPE = (64, 64)
N_IMAGES = 200
N_SAMPLES = 8
N_BINS = 16


def build(noise_scale: float = 1.0, mean_scale: float = 1.0, seed: int = 555):
    """Run the toy end to end with an optionally miscalibrated sampler.

    Args:
        noise_scale (float): Multiplier on the posterior standard deviation.
        mean_scale (float): Multiplier on the Wiener posterior mean.
        seed (int): Seed for targets, observations, and samples.

    Returns:
        dict: Calibration ratios, fill curves, binning, and the toy model.
    """
    rng = np.random.default_rng(seed)
    model = toy.make_diagonal_posterior(SHAPE, prior_exponent=2.0, blur_sigma=3.0)
    x = model.draw_targets(N_IMAGES, rng)
    z = model.encode(x, rng)
    y = model.sample(z, N_SAMPLES, rng, noise_scale=noise_scale, mean_scale=mean_scale)

    setup = sf.make_setup(SHAPE, "none")
    binning = sf.radial_binning(setup, N_BINS, f_max=0.5)
    x_hat = sf.preprocess(x, setup)
    y_hat = sf.preprocess(y.reshape(-1, *SHAPE), setup).reshape(N_IMAGES, N_SAMPLES, *SHAPE)
    xy = sf.per_image_bin_sums(x_hat, y_hat[:, 0], binning, setup)
    yy = sf.sample_pair_spectra(y_hat, binning, setup)
    return {
        "model": model,
        "setup": setup,
        "binning": binning,
        "xy": xy,
        "yy": yy,
        "cal": sf.calibration_ratios(xy, yy, min_shared=0.05),
        "fill": sf.fill_curves(sf.accumulate_fill(x_hat, y_hat, binning, setup)),
    }


@pytest.fixture(scope="module")
def calibrated():
    """Reference run with a correctly calibrated sampler."""
    return build()


@pytest.fixture(scope="module")
def under_dispersed():
    """Run with the posterior noise halved, an over-confident sampler."""
    return build(noise_scale=0.5)


@pytest.fixture(scope="module")
def over_dispersed():
    """Run with the posterior noise doubled, an under-confident sampler."""
    return build(noise_scale=2.0)


@pytest.fixture(scope="module")
def over_guided():
    """Run with the posterior mean inflated, the stand-in for too-strong guidance."""
    return build(mean_scale=1.3)


def _predicted_power_ratio(model, binning, setup, noise_scale):
    """Predict S_YY/S_XX per bin for a noise-scaled sampler.

    Args:
        model (toy.DiagonalPosterior): The toy model.
        binning (sf.Binning): Bin assignment.
        setup (sf.SpectralSetup): Geometry.
        noise_scale (float): Multiplier applied to the posterior standard deviation.

    Returns:
        numpy.ndarray: Predicted ``S_YY / S_XX`` per bin.
    """
    p_bar = sf.bin_mean_of_modes(model.prior_spectrum, binning, setup)
    v_bar = sf.bin_mean_of_modes(model.posterior_variance, binning, setup)
    return (p_bar + (noise_scale**2 - 1.0) * v_bar) / p_bar


def test_calibrated_reference_passes_both_diagnostics(calibrated):
    """The reference run sits on one for both ratios, so deviations mean something."""
    cal = calibrated["cal"]
    np.testing.assert_allclose(cal["ratio_power"], 1.0, atol=0.07)
    np.testing.assert_allclose(cal["ratio_cross"][cal["reliable"]], 1.0, atol=0.1)


@pytest.mark.parametrize("name,scale", [("under_dispersed", 0.5), ("over_dispersed", 2.0)])
def test_noise_miscalibration_is_detected_in_the_right_direction(request, name, scale):
    """Halved noise pushes the power ratio below one; doubled noise pushes it above."""
    run = request.getfixturevalue(name)
    ratio = run["cal"]["ratio_power"]
    saturated = run["fill"]["fill_fraction"] > 0.5
    if scale < 1.0:
        assert np.all(ratio[saturated] < 0.98), "under-dispersion must show as a power deficit"
        assert np.median(run["cal"]["log2_ratio_power"][saturated]) < -0.5
    else:
        assert np.all(ratio[saturated] > 1.02), "over-dispersion must show as a power excess"
        assert np.median(run["cal"]["log2_ratio_power"][saturated]) > 0.5


@pytest.mark.parametrize("name,scale", [("under_dispersed", 0.5), ("over_dispersed", 2.0)])
def test_noise_miscalibration_matches_its_analytic_size(request, name, scale):
    """The power ratio does not merely move, it moves by P + (c^2-1)V over P."""
    run = request.getfixturevalue(name)
    predicted = _predicted_power_ratio(run["model"], run["binning"], run["setup"], scale)
    np.testing.assert_allclose(run["cal"]["ratio_power"], predicted, rtol=0.07)


@pytest.mark.parametrize("name", ["under_dispersed", "over_dispersed"])
def test_cross_ratio_is_blind_to_noise_miscalibration(request, name):
    """Scaling only the posterior noise leaves S_XY and S_{Y_j Y_k} both equal to P-V.

    This is the reason the report must show both ratios.  A sampler whose spread
    is wrong but whose mean is right passes the cross-ratio test completely.
    """
    cal = request.getfixturevalue(name)["cal"]
    reliable = cal["reliable"]
    assert reliable.sum() >= 3
    np.testing.assert_allclose(cal["ratio_cross"][reliable], 1.0, atol=0.1)


def test_guidance_miscalibration_is_detected_by_the_cross_ratio(over_guided):
    """Inflating the posterior mean by g drives the cross ratio to 1/g."""
    cal = over_guided["cal"]
    reliable = cal["reliable"]
    assert reliable.sum() >= 3
    np.testing.assert_allclose(cal["ratio_cross"][reliable], 1.0 / 1.3, rtol=0.08)
    assert np.all(cal["ratio_cross"][reliable] < 0.95)


def test_guidance_miscalibration_also_shows_as_excess_power(over_guided):
    """An inflated mean adds power too, so the two diagnostics disagree in a readable way."""
    cal = over_guided["cal"]
    determined = cal["shared_fraction"] > 0.2
    assert np.all(cal["ratio_power"][determined] > 1.02)


def test_the_two_failure_modes_are_distinguishable(under_dispersed, over_guided):
    """Noise and gain errors leave different signatures, so the pair identifies the cause."""
    noise_cross = under_dispersed["cal"]["ratio_cross"][under_dispersed["cal"]["reliable"]]
    gain_cross = over_guided["cal"]["ratio_cross"][over_guided["cal"]["reliable"]]
    assert abs(np.median(noise_cross) - 1.0) < 0.1
    assert abs(np.median(gain_cross) - 1.0) > 0.15
