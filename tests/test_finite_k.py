"""Test 6: how the estimates converge as K grows, and what K to use on the real run.

Two quantities behave very differently in K, and the recommendation has to respect
both:

* The **fill fraction** uses Bessel's correction, so it is unbiased for every
  ``K >= 2``.  What improves with K is its variance, not its centre.
* The **sample-mean coherence** is biased **downward** at finite K, because the
  denominator ``S_{Ybar Ybar} = |mu|^2 + V / K`` is inflated by the sample mean's
  own noise.  The bias decays as ``1 / K``, which is slow, and it is largest
  exactly where the fill fraction is largest.  The split-half estimator removes
  it, at the cost of using half the samples in each half-mean.

The sweep is written to disk so the K chosen for the real run can be justified
from the curve rather than asserted.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import analysis.spectral_fillin as sf
import toy
from conftest import ARTIFACT_DIR

SHAPE = (48, 48)
N_IMAGES = 128
K_MAX = 32
K_VALUES = [2, 3, 4, 6, 8, 12, 16, 24, 32]
N_BINS = 12


@pytest.fixture(scope="module")
def sweep():
    """Draw K_MAX samples once, then recompute every estimate on nested subsets.

    Using nested subsets of a single draw rather than independent draws per K
    isolates the effect of K itself: the targets, the latents, and the first K
    samples are identical across the sweep.

    Returns:
        dict: Per-K curves, the analytic limits, and the shared binning.
    """
    rng = np.random.default_rng(20240608)
    # A mild blur with a high noise floor spreads the determined-to-filled
    # transition across the whole band.  A sharper forward model compresses it
    # into one or two bins, where "max relative bias over informative bins"
    # degenerates into a statement about those bins and says nothing useful
    # about the K needed on the real run.
    model = toy.make_diagonal_posterior(
        SHAPE, prior_exponent=2.0, blur_sigma=1.0, noise_power=4.7
    )
    x = model.draw_targets(N_IMAGES, rng)
    z = model.encode(x, rng)
    y = model.sample(z, K_MAX, rng)

    setup = sf.make_setup(SHAPE, "none")
    binning = sf.radial_binning(setup, N_BINS, f_max=0.5)
    x_hat = sf.preprocess(x, setup)
    y_hat = sf.preprocess(y.reshape(-1, *SHAPE), setup).reshape(N_IMAGES, K_MAX, *SHAPE)

    p_bar = sf.bin_mean_of_modes(model.prior_spectrum, binning, setup)
    v_bar = sf.bin_mean_of_modes(model.posterior_variance, binning, setup)
    determined = 1.0 - v_bar / p_bar

    rows = {"K": [], "fill": [], "fill_se": [], "gamma2_naive": [], "gamma2_debiased": [], "gamma2_single": []}
    for k in K_VALUES:
        subset = y_hat[:, :k]
        acc = sf.accumulate_fill(x_hat, subset, binning, setup)
        curves = sf.fill_curves(acc)
        ci = sf.bootstrap_ci(
            lambda idx: sf.fill_curves(acc, idx)["fill_fraction"],
            n_images=N_IMAGES,
            n_boot=120,
            seed=42,
        )
        naive = sf.reduce_bin_sums(
            sf.per_image_bin_sums(x_hat, subset.mean(axis=1), binning, setup)
        )["gamma2"]
        if k >= 4:
            split = sf.split_half_mean_spectra(x_hat, subset, binning, setup, seed=1)
            debiased = sf.debiased_mean_coherence(split)["gamma2_debiased"]
        else:
            debiased = np.full(binning.n_bins, np.nan)
        rows["K"].append(k)
        rows["fill"].append(curves["fill_fraction"])
        rows["fill_se"].append(ci["std"])
        rows["gamma2_naive"].append(naive)
        rows["gamma2_debiased"].append(debiased)
        rows["gamma2_single"].append(
            sf.reduce_bin_sums(sf.per_image_bin_sums(x_hat, subset[:, 0], binning, setup))["gamma2"]
        )

    out = {key: np.asarray(value) for key, value in rows.items()}
    out["binning"] = binning
    out["f_centers"] = binning.f_centers
    out["fill_limit"] = v_bar / p_bar
    out["gamma2_mean_limit"] = determined
    # Analytic naive curve: the denominator carries an extra V/K per mode.
    out["gamma2_naive_predicted"] = np.stack(
        [(p_bar - v_bar) ** 2 / (p_bar * (p_bar - v_bar + v_bar / k)) for k in K_VALUES]
    )
    return out


def test_fill_fraction_is_unbiased_at_every_k(sweep):
    """Bessel's correction makes the fill fraction centre-correct even at K=2."""
    for i, k in enumerate(K_VALUES):
        np.testing.assert_allclose(
            sweep["fill"][i], sweep["fill_limit"], atol=0.04,
            err_msg=f"fill fraction is off at K={k}",
        )


def test_fill_fraction_precision_improves_with_k(sweep):
    """What K buys for the fill fraction is precision, and it scales as 1/sqrt(K-1)."""
    se = sweep["fill_se"].mean(axis=1)
    assert se[0] > se[-1]
    assert np.all(np.diff(se) <= 1e-6), "standard error must fall monotonically in K"
    ratio = se[0] / se[K_VALUES.index(16)]
    assert ratio > 2.0, f"K=2 to K=16 should tighten the interval several-fold, got {ratio:.2f}"


def test_naive_mean_coherence_is_biased_downward_and_converges(sweep):
    """The naive sample-mean coherence rises monotonically toward its limit in K."""
    naive = sweep["gamma2_naive"]
    informative = sweep["gamma2_mean_limit"] > 0.05
    for i in range(len(K_VALUES) - 1):
        assert np.all(naive[i + 1][informative] >= naive[i][informative] - 5e-3)
    assert np.all(naive[0][informative] < sweep["gamma2_mean_limit"][informative])


def test_naive_mean_coherence_matches_its_analytic_bias(sweep):
    """The downward bias is exactly the V/K inflation of the denominator."""
    np.testing.assert_allclose(sweep["gamma2_naive"], sweep["gamma2_naive_predicted"], atol=0.03)


def test_split_half_estimator_removes_the_finite_k_bias(sweep):
    """The debiased estimate sits on its limit for every K where it is defined."""
    for i, k in enumerate(K_VALUES):
        if k < 4:
            assert np.all(np.isnan(sweep["gamma2_debiased"][i]))
            continue
        np.testing.assert_allclose(
            sweep["gamma2_debiased"][i], sweep["gamma2_mean_limit"], atol=0.05,
            err_msg=f"debiased mean coherence is off at K={k}",
        )


def test_sample_mean_beats_a_single_sample(sweep):
    """Averaging samples recovers more about X than any one sample does."""
    informative = sweep["gamma2_mean_limit"] > 0.05
    for i, k in enumerate(K_VALUES):
        if k < 4:
            continue
        assert np.all(
            sweep["gamma2_debiased"][i][informative] >= sweep["gamma2_single"][i][informative] - 1e-6
        )


def test_sweep_is_saved_with_a_k_recommendation(sweep):
    """Persist the sweep and derive K from fill-fraction precision, not from the naive bias.

    With a forward model whose SNR falls gradually across the band, the naive
    sample-mean coherence is still biased by tens of percent at K=32.  That is not
    a reason to raise K: the bias is ``(V/K) / (S_mumu + V/K)``, and wherever
    fill-in dominates, ``S_mumu`` is small enough that no affordable K rescues it.
    The split-half estimator removes the bias outright and is flat from K=4
    upward, so it is the estimator the real run should use, and K is then chosen
    purely for the precision of the fill fraction.
    """
    limit = sweep["gamma2_mean_limit"]
    informative = limit > 0.05
    relative_bias = np.array(
        [np.max((limit[informative] - row[informative]) / limit[informative])
         for row in sweep["gamma2_naive"]]
    )
    standard_error = sweep["fill_se"].mean(axis=1)
    normalized = standard_error / standard_error[-1]
    # Smallest K whose fill-fraction standard error is within 20 percent of the
    # K=32 value, floored at 4 so the split-half estimator is available.
    adequate = [k for k, value in zip(K_VALUES, normalized) if value <= 1.2 and k >= 4]
    recommended = int(adequate[0]) if adequate else int(K_VALUES[-1])

    np.savez(
        ARTIFACT_DIR / "finite_k_sweep.npz",
        K=np.asarray(K_VALUES),
        f_centers=sweep["f_centers"],
        fill=sweep["fill"],
        fill_se=sweep["fill_se"],
        fill_limit=sweep["fill_limit"],
        gamma2_naive=sweep["gamma2_naive"],
        gamma2_naive_predicted=sweep["gamma2_naive_predicted"],
        gamma2_debiased=sweep["gamma2_debiased"],
        gamma2_single=sweep["gamma2_single"],
        gamma2_mean_limit=sweep["gamma2_mean_limit"],
        relative_bias=relative_bias,
    )
    debiased_error = [
        float(np.nanmax(np.abs(row[informative] - limit[informative]))) if k >= 4 else None
        for k, row in zip(K_VALUES, sweep["gamma2_debiased"])
    ]
    summary = {
        "k_values": K_VALUES,
        "fill_standard_error": standard_error.tolist(),
        "fill_standard_error_normalized_to_k32": normalized.tolist(),
        "naive_mean_coherence_max_relative_bias": relative_bias.tolist(),
        "debiased_mean_coherence_max_absolute_error": debiased_error,
        "recommended_k": recommended,
        "recommendation_basis": (
            "Smallest K within 20 percent of the K=32 fill-fraction standard error, "
            "floored at 4 so the split-half debiased estimator is available."
        ),
        "warning_about_naive_estimator": (
            "The naive X-versus-Ybar coherence is still biased low by "
            f"{relative_bias[-1]:.0%} at K=32 in bins where fill-in dominates. Its bias is "
            "(V/K)/(S_mumu + V/K), which no affordable K removes once S_mumu is small. "
            "Use the split-half debiased estimator instead; it is flat from K=4 upward."
        ),
    }
    (ARTIFACT_DIR / "finite_k_summary.json").write_text(json.dumps(summary, indent=2))
    assert (ARTIFACT_DIR / "finite_k_sweep.npz").exists()
    assert relative_bias[0] > relative_bias[-1]
    assert recommended >= 4


def test_debiased_estimator_beats_the_naive_one_at_every_k(sweep):
    """The split-half estimate is closer to the truth than the naive one, for all K>=4."""
    limit = sweep["gamma2_mean_limit"]
    informative = limit > 0.05
    for i, k in enumerate(K_VALUES):
        if k < 4:
            continue
        naive_error = np.max(np.abs(sweep["gamma2_naive"][i][informative] - limit[informative]))
        debiased_error = np.max(np.abs(sweep["gamma2_debiased"][i][informative] - limit[informative]))
        assert debiased_error < naive_error, f"debiasing did not help at K={k}"

def test_finite_k_figure_is_written(sweep):
    """Save the convergence figure that justifies the K chosen for the real run."""
    plt = pytest.importorskip("matplotlib.pyplot")
    f = sweep["f_centers"]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    for i, k in enumerate(K_VALUES):
        shade = plt.cm.viridis(i / (len(K_VALUES) - 1))
        axes[0].plot(f, sweep["fill"][i], color=shade, lw=1.2, label=f"K={k}")
        axes[1].plot(f, sweep["gamma2_naive"][i], color=shade, lw=1.2, label=f"K={k}")
    axes[0].plot(f, sweep["fill_limit"], "k--", lw=2, label="exact")
    axes[0].set_title("Fill fraction is unbiased at every K")
    axes[1].plot(f, sweep["gamma2_debiased"][K_VALUES.index(4)], "r-", lw=1.6,
                 label="split-half debiased, K=4")
    axes[1].plot(f, sweep["gamma2_mean_limit"], "k--", lw=2, label="K $\\to\\infty$ limit")
    axes[1].set_title("Naive is biased down as 1/K; split-half is not")

    informative = sweep["gamma2_mean_limit"] > 0.05
    bias = [
        np.max((sweep["gamma2_mean_limit"][informative] - row[informative])
               / sweep["gamma2_mean_limit"][informative])
        for row in sweep["gamma2_naive"]
    ]
    axes[2].plot(K_VALUES, bias, "o-", label="naive, max relative bias")
    standard_error = sweep["fill_se"].mean(axis=1)
    axes[2].plot(K_VALUES, standard_error / standard_error[-1], "s-",
                 label="fill fraction SE (relative to K=32)")
    axes[2].axhline(1.2, color="C1", ls="--", lw=1.0, label="within 20% of K=32")
    axes[2].axhline(0.05, color="k", ls=":", lw=1.2, label="5% threshold")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("K (samples per latent)")
    axes[2].set_title("Choosing K")

    for axis in axes[:2]:
        axis.set_xlabel("spatial frequency (cycles/pixel)")
    axes[0].set_ylabel("fill fraction")
    axes[1].set_ylabel("$\\gamma^2(X, \\bar{Y})$")
    for axis in axes:
        axis.legend(fontsize=6.5)
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "finite_k_convergence.png", dpi=150)
    plt.close(figure)
    assert (ARTIFACT_DIR / "finite_k_convergence.png").exists()
