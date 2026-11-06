"""Figures for the RGC spectral coherence and fill-in analysis.

Every function takes the output of a :mod:`analysis.pipeline` stage and returns a
matplotlib figure, so the notebook can show them inline and the headless script
can save the same figures without duplicating any logic.

Overlay convention.  Each mosaic contributes its own Nyquist line, because the
four mosaics differ in spacing by a factor of three and a single line would be
misleading.  On radial plots the line is the Brillouin-zone inradius, the
conservative scalar below which no orientation aliases.  On orientation-resolved
plots the full direction-dependent boundary is drawn.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from . import spectral_fillin as sf

__all__ = [
    "MOSAIC_COLORS",
    "add_nyquist_lines",
    "add_content_limit",
    "add_cycles_per_degree_axis",
    "plot_spectra",
    "plot_coherence",
    "plot_information",
    "plot_fill_curves",
    "plot_error_decomposition",
    "plot_calibration",
    "plot_oriented",
    "plot_pixel_maps",
    "plot_examples",
    "plot_per_image_scores",
    "plot_loss_traces",
]

MOSAIC_COLORS = {
    "ON_Parasol": "#d62728",
    "OFF_Parasol": "#ff7f0e",
    "ON_Midget": "#2ca02c",
    "OFF_Midget": "#1f77b4",
    "ON_Smooth_Monostratified": "#9467bd",
}


def add_nyquist_lines(axis: plt.Axes, nyquist: dict[str, dict[str, float]],
                      label: bool = True, band: bool = True) -> None:
    """Draw one Nyquist marker per mosaic on a frequency axis.

    The line is drawn at the Brillouin-zone inradius and, when ``band`` is set, a
    shaded strip extends to the circumradius.  The strip is the honest picture: a
    hexagonal mosaic does not have one Nyquist frequency, it has a range spanning
    a factor of ``2/sqrt(3)`` depending on grating orientation.

    Args:
        axis (matplotlib.axes.Axes): Axis to annotate.
        nyquist (dict): Output of :func:`analysis.pipeline.nyquist_table`.
        label (bool): Whether to add legend entries.
        band (bool): Whether to shade from inradius to circumradius.
    """
    for name, entry in sorted(nyquist.items(), key=lambda kv: kv[1]["nyquist_min"]):
        color = MOSAIC_COLORS.get(name, "0.4")
        axis.axvline(entry["nyquist_min"], color=color, ls=":", lw=1.4,
                     label=f"{name} Nyquist" if label else None)
        if band:
            axis.axvspan(entry["nyquist_min"], entry["nyquist_max"], color=color, alpha=0.06, lw=0)


def add_cycles_per_degree_axis(axis: plt.Axes, microns_per_pixel: float | None,
                               microns_per_degree: float | None) -> plt.Axes | None:
    """Add a secondary cycles-per-degree axis when the retinal scale is known.

    The repository's YAML encodes receptive-field sizes as ``microns / 30``, which
    implies 30 microns per pixel, but it never records microns per degree, so the
    conversion cannot be inferred from the configuration alone and the axis is
    omitted unless both numbers are supplied.

    Args:
        axis (matplotlib.axes.Axes): Primary axis in cycles per pixel.
        microns_per_pixel (float | None): Retinal microns per image pixel.
        microns_per_degree (float | None): Retinal microns per degree of visual angle.

    Returns:
        matplotlib.axes.Axes | None: The secondary axis, or ``None`` if omitted.
    """
    if not microns_per_pixel or not microns_per_degree:
        return None
    factor = microns_per_degree / microns_per_pixel
    secondary = axis.secondary_xaxis("top", functions=(lambda v: v * factor, lambda v: v / factor))
    secondary.set_xlabel("spatial frequency (cycles/degree)")
    return secondary


def add_content_limit(axis: plt.Axes, content_limit: float | None, label: bool = True) -> None:
    """Shade the band above which the targets carry no content of their own.

    When targets are resampled up from smaller sources they are band-limited, and
    above that limit ``S_XX`` measures resampling, not the scene.  Any ratio with
    ``S_XX`` in it, in particular the power-ratio calibration diagnostic, is
    therefore uninterpretable there and the band is greyed out rather than read.

    Args:
        axis (matplotlib.axes.Axes): Axis to annotate.
        content_limit (float | None): Limit in cycles/pixel, or ``None`` to skip.
        label (bool): Whether to add a legend entry.
    """
    if not content_limit or content_limit >= 0.5:
        return
    axis.axvspan(content_limit, 0.5, color="0.5", alpha=0.18, lw=0,
                 label="targets band-limited" if label else None)


def _frequency_axis(axis: plt.Axes, nyquist: dict | None, label: bool = True,
                    microns_per_pixel: float | None = None,
                    microns_per_degree: float | None = None,
                    content_limit: float | None = None) -> None:
    """Apply the shared frequency-axis styling and overlays.

    Args:
        axis (matplotlib.axes.Axes): Axis to style.
        nyquist (dict | None): Mosaic Nyquist table, or ``None`` to skip overlays.
        label (bool): Whether Nyquist lines contribute legend entries.
        microns_per_pixel (float | None): Retinal microns per pixel.
        microns_per_degree (float | None): Retinal microns per degree.
        content_limit (float | None): Target bandwidth limit in cycles/pixel.
    """
    axis.set_xlabel("spatial frequency (cycles/pixel)")
    if nyquist:
        add_nyquist_lines(axis, nyquist, label=label)
    add_content_limit(axis, content_limit, label=label)
    add_cycles_per_degree_axis(axis, microns_per_pixel, microns_per_degree)


def plot_spectra(coherence: dict[str, Any], nyquist: dict | None = None, **scale) -> plt.Figure:
    """Plot the target and reconstruction power spectra on log axes.

    Args:
        coherence (dict): Output of :func:`analysis.pipeline.run_coherence`.
        nyquist (dict | None): Mosaic Nyquist table for the overlay.
        **scale: Optional ``microns_per_pixel`` and ``microns_per_degree``.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    f = coherence["f_centers"]
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.loglog(f, coherence["single"]["S_XX"], "o-", color="k", label="$S_{XX}$ target")
    axis.loglog(f, coherence["single"]["S_YY"], "s-", color="C0", label="$S_{YY}$ reconstruction")
    axis.loglog(f, np.abs(coherence["single"]["S_XY"]), "^-", color="C2", label="$|S_{XY}|$")
    axis.loglog(f, np.abs(coherence["sample_pair"]["S_XY"]), "v-", color="C3",
                label="$S_{Y_jY_k}$ shared between samples")
    _frequency_axis(axis, nyquist, **scale)
    axis.set_ylabel("power (arbitrary units)")
    axis.set_title("Power and cross spectra")
    axis.legend(fontsize=7.5)
    figure.tight_layout()
    return figure


def plot_coherence(coherence: dict[str, Any], nyquist: dict | None = None, **scale) -> plt.Figure:
    """Plot the three coherence variants with intervals and the shuffle null.

    Args:
        coherence (dict): Output of :func:`analysis.pipeline.run_coherence`.
        nyquist (dict | None): Mosaic Nyquist table for the overlay.
        **scale: Optional ``microns_per_pixel`` and ``microns_per_degree``.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    f = coherence["f_centers"]
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    series = [
        ("single", "C0", "o", "$X$ vs single sample $Y_1$"),
        ("mean", "C1", "s", r"$X$ vs sample mean $\bar{Y}$ (naive)"),
        ("sample_pair", "C2", "^", "$Y_j$ vs $Y_k$"),
    ]
    for key, color, marker, label in series:
        entry = coherence[key]
        axes[0].fill_between(f, entry["ci_lo"], entry["ci_hi"], color=color, alpha=0.2)
        axes[0].plot(f, entry["gamma2"], marker + "-", color=color, ms=4, label=label)
    if coherence.get("mean_debiased"):
        entry = coherence["mean_debiased"]
        axes[0].fill_between(f, entry["ci_lo"], entry["ci_hi"], color="C3", alpha=0.2)
        axes[0].plot(f, entry["gamma2"], "d-", color="C3", ms=4,
                     label=r"$X$ vs $\bar{Y}$ (split-half debiased)")
    axes[0].plot(f, coherence["single"]["gamma2_null"], "k:", lw=1.2, label="shuffle null")
    axes[0].set_ylabel(r"coherence $\gamma^2$")
    axes[0].set_title("Coherence (shaded: 95% CI, scene-clustered bootstrap)")
    axes[0].set_ylim(-0.03, 1.03)

    axes[1].plot(f, coherence["single"]["gamma2"], "o-", color="C0", ms=4, label="raw")
    axes[1].plot(f, coherence["single"]["gamma2_corrected"], "s-", color="C4", ms=4,
                 label="shuffle-corrected")
    axes[1].plot(f, coherence["single"]["gamma2_null"], "k:", lw=1.2, label="null")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"coherence $\gamma^2$ (log)")
    axes[1].set_title("Single-sample coherence: raw, null, corrected")

    for axis in axes:
        _frequency_axis(axis, nyquist, label=axis is axes[0], **scale)
        axis.legend(fontsize=7)
    figure.tight_layout()
    return figure


def plot_information(coherence: dict[str, Any], nyquist: dict | None = None, **scale) -> plt.Figure:
    """Plot the information lower bound per bin and its cumulative total.

    Args:
        coherence (dict): Output of :func:`analysis.pipeline.run_coherence`.
        nyquist (dict | None): Mosaic Nyquist table for the overlay.
        **scale: Optional ``microns_per_pixel`` and ``microns_per_degree``.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    f = coherence["f_centers"]
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    for key, color, label in [("single", "C0", "single sample"), ("mean", "C1", "sample mean")]:
        axes[0].plot(f, coherence[key]["bits"], "o-", color=color, ms=4, label=label)
        axes[1].plot(f, np.nancumsum(coherence[key]["bits"]), "-", color=color, label=label)
    axes[0].plot(f, coherence["single"]["bits_corrected"], "s--", color="C4", ms=4,
                 label="single, shuffle-corrected")
    axes[1].plot(f, np.nancumsum(coherence["single"]["bits_corrected"]), "--", color="C4",
                 label="single, shuffle-corrected")

    axes[0].set_ylabel("information (bits per bin)")
    axes[0].set_title("Information lower bound per frequency bin")
    axes[1].set_ylabel("cumulative information (bits/image)")
    axes[1].set_title("Cumulative information (Gaussian lower bound)")
    for axis in axes:
        _frequency_axis(axis, nyquist, label=axis is axes[0], **scale)
        axis.legend(fontsize=7.5)
    figure.tight_layout()
    return figure


def plot_fill_curves(fill: dict[str, Any], nyquist: dict | None = None, **scale) -> plt.Figure:
    """Plot the fill fraction, its target-normalized version, and determined fraction.

    Args:
        fill (dict): Output of :func:`analysis.pipeline.run_fill`.
        nyquist (dict | None): Mosaic Nyquist table for the overlay.
        **scale: Optional ``microns_per_pixel`` and ``microns_per_degree``.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    f = fill["f_centers"]
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    axes[0].fill_between(f, fill["fill_ci_lo"], fill["fill_ci_hi"], color="C0", alpha=0.25)
    axes[0].plot(f, fill["fill_fraction"], "o-", color="C0", ms=4, label="$F = V_{fill}/S_{YY}$")
    axes[0].plot(f, fill["determined_fraction"], "s--", color="C1", ms=4, label="determined $1-F$")
    axes[0].plot(f, fill["fill_fraction_vs_xx"], "^:", color="C2", ms=4, label="$V_{fill}/S_{XX}$")
    axes[0].axhline(0.5, color="0.6", lw=0.8)
    axes[0].set_ylabel("fraction of power")
    axes[0].set_title("Fill-in: what the prior re-draws per frequency")
    axes[0].set_ylim(-0.05, 1.3)

    axes[1].loglog(f, fill["S_XX"], "o-", color="k", label="$S_{XX}$")
    axes[1].loglog(f, fill["S_YY"], "s-", color="C0", label="$S_{YY}$")
    axes[1].loglog(f, fill["V_fill"], "^-", color="C3", label="$V_{fill}$")
    axes[1].loglog(f, np.abs(fill["total_variance"]["explained"]), "v-", color="C2",
                   label=r"$\mathrm{Var}_n(E[\hat{Y}|Z])$")
    axes[1].set_ylabel("power (arbitrary units)")
    axes[1].set_title("Law of total variance: $S_{YY}$ = explained + fill-in")

    for axis in axes:
        _frequency_axis(axis, nyquist, label=axis is axes[0], **scale)
        axis.legend(fontsize=7.5)
    figure.tight_layout()
    return figure


def plot_error_decomposition(fill: dict[str, Any], nyquist: dict | None = None, **scale) -> plt.Figure:
    """Plot the squared-error split into bias and fill-in terms.

    Args:
        fill (dict): Output of :func:`analysis.pipeline.run_fill`.
        nyquist (dict | None): Mosaic Nyquist table for the overlay.
        **scale: Optional ``microns_per_pixel`` and ``microns_per_degree``.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    f = fill["f_centers"]
    decomposition = fill["error_decomposition"]
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    axes[0].loglog(f, decomposition["lhs"], "k-", lw=2, label=r"$E|\hat{X}-\hat{Y}_k|^2$")
    axes[0].loglog(f, decomposition["bias_term"], "o-", color="C1", ms=4,
                   label=r"$|\hat{X}-\bar{Y}|^2$ (bias)")
    axes[0].loglog(f, decomposition["fill_term"], "^-", color="C3", ms=4,
                   label=r"$\frac{K-1}{K}V_{fill}$ (fill-in)")
    axes[0].set_ylabel("squared error (arbitrary units)")
    axes[0].set_title("Error decomposition")

    axes[1].plot(f, fill["observed_error_share"], "o-", color="C3", ms=4, label="observed share")
    axes[1].axhline(fill["calibrated_error_share"], color="k", ls="--", lw=1.4,
                    label=r"$(K-1)/2K$: exact-posterior value")
    axes[1].set_ylabel("fill-in share of squared error")
    axes[1].set_title("Exchangeability check: share should be flat at $(K-1)/2K$")
    axes[1].set_ylim(0, 1)

    for axis in axes:
        _frequency_axis(axis, nyquist, label=axis is axes[0], **scale)
        axis.legend(fontsize=7.5)
    figure.tight_layout()
    return figure


def plot_calibration(calibration: dict[str, Any], nyquist: dict | None = None, **scale) -> plt.Figure:
    """Plot the two exchangeability ratios against frequency with intervals.

    Args:
        calibration (dict): Output of :func:`analysis.pipeline.run_calibration`.
        nyquist (dict | None): Mosaic Nyquist table for the overlay.
        **scale: Optional ``microns_per_pixel`` and ``microns_per_degree``.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    f = calibration["f_centers"]
    reliable = calibration["reliable"]
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    power_ci = calibration["ratio_power_ci"]
    axes[0].fill_between(f, power_ci["lo"], power_ci["hi"], color="C0", alpha=0.25)
    axes[0].plot(f, calibration["ratio_power"], "o-", color="C0", ms=4)
    axes[0].axhline(1.0, color="k", ls="--", lw=1.4, label="exchangeable")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$S_{YY} / S_{XX}$")
    axes[0].set_title("Power ratio: <1 too smooth, >1 too much injected")

    cross_ci = calibration["ratio_cross_ci"]
    axes[1].fill_between(f[reliable], cross_ci["lo"][reliable], cross_ci["hi"][reliable],
                         color="C3", alpha=0.25)
    axes[1].plot(f[reliable], calibration["ratio_cross_raw"][reliable], "o-", color="C3", ms=4,
                 label="reliable bins")
    axes[1].plot(f[~reliable], calibration["ratio_cross_raw"][~reliable], "x", color="0.7", ms=5,
                 label="nothing shared: uninterpretable")
    axes[1].axhline(1.0, color="k", ls="--", lw=1.4, label="exchangeable")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$|S_{XY}| / S_{Y_jY_k}$")
    axes[1].set_title("Cross ratio: <1 means samples agree with each other, not the truth")

    for axis in axes:
        _frequency_axis(axis, nyquist, label=axis is axes[0], **scale)
        axis.legend(fontsize=7)
    figure.tight_layout()
    return figure


def plot_oriented(oriented: dict[str, Any], spacings: dict[str, float] | None = None) -> plt.Figure:
    """Plot coherence and fill fraction resolved by orientation, in polar form.

    The hexagonal Brillouin-zone boundary is drawn for each mosaic, which is where
    the direction dependence of a hexagonal mosaic becomes visible: the limit
    varies by a factor of ``2/sqrt(3)`` between the zone faces and its corners.

    Args:
        oriented (dict): Output of :func:`analysis.pipeline.run_oriented`.
        spacings (dict[str, float] | None): Mosaic lattice spacings in pixels.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    f = oriented["f_centers"]
    theta = oriented["theta_centers"]
    # Duplicate the half-plane onto the full circle: theta and theta+pi describe
    # the same oriented grating, so the polar map is two-fold symmetric.
    theta_full = np.concatenate([theta, theta + np.pi, [theta[0] + 2 * np.pi]])

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), subplot_kw={"projection": "polar"})
    panels = [("gamma2", r"coherence $\gamma^2$ (single sample)", "viridis", (0, 1)),
              ("fill_fraction", "fill fraction $F$", "magma", (0, 1))]
    for axis, (key, title, cmap, limits) in zip(axes, panels):
        grid = oriented[key]
        wrapped = np.concatenate([grid, grid, grid[:, :1]], axis=1)
        mesh = axis.pcolormesh(theta_full, f, wrapped, cmap=cmap, vmin=limits[0], vmax=limits[1],
                               shading="nearest")
        axis.set_title(title, pad=16)
        axis.set_rlabel_position(135)
        figure.colorbar(mesh, ax=axis, pad=0.12, shrink=0.75)
        if spacings:
            fine = np.linspace(0, 2 * np.pi, 361)
            for name, spacing in sorted(spacings.items(), key=lambda kv: -kv[1]):
                axis.plot(fine, sf.hex_nyquist(float(spacing), fine), lw=1.3,
                          color=MOSAIC_COLORS.get(name, "w"), label=name)
        axis.set_rmax(f.max())
    axes[0].legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(-0.18, -0.12))
    figure.suptitle("Orientation-resolved analysis, with hexagonal Brillouin-zone boundaries")
    figure.tight_layout()
    return figure


def plot_pixel_maps(fill: dict[str, Any]) -> plt.Figure:
    """Plot the mean within-Z variance map and the distribution over images.

    Args:
        fill (dict): Output of :func:`analysis.pipeline.run_fill`.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    maps = fill["pixel_maps"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    image = axes[0].imshow(maps["mean"], cmap="magma")
    axes[0].set_title(r"mean over images of $\mathrm{Var}(Y|Z)$ per pixel")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    figure.colorbar(image, ax=axes[0], shrink=0.8)

    axes[1].hist(np.sqrt(maps["per_image_scalar"]), bins=24, color="C3", alpha=0.85)
    axes[1].set_xlabel(r"per-image RMS of $\mathrm{Var}(Y|Z)^{1/2}$ (grey levels)")
    axes[1].set_ylabel("images")
    axes[1].set_title("How much the prior re-draws, per image")
    figure.tight_layout()
    return figure


def plot_examples(bank, fill: dict[str, Any], indices: list[int]) -> plt.Figure:
    """Show target, sample mean, two samples, and the within-Z variance map.

    Args:
        bank: The loaded :class:`analysis.pipeline.Bank`.
        fill (dict): Output of :func:`analysis.pipeline.run_fill`.
        indices (list[int]): Image indices to display, one per row.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    maps = fill["pixel_maps"]
    columns = ["target $X$", r"sample mean $\bar{Y}$", "sample $Y_1$", "sample $Y_2$",
               r"$\mathrm{Var}(Y|Z)$"]
    figure, axes = plt.subplots(len(indices), 5, figsize=(13.0, 2.7 * len(indices)))
    axes = np.atleast_2d(axes)
    for row, index in enumerate(indices):
        panels = [bank.x[index], maps["sample_mean"][index], bank.y[index, 0], bank.y[index, 1]]
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel, cmap="gray", vmin=0, vmax=1)
        variance = axes[row, 4].imshow(maps["per_image"][index], cmap="magma")
        figure.colorbar(variance, ax=axes[row, 4], shrink=0.82)
        axes[row, 0].set_ylabel(str(bank.labels[index]), fontsize=7)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    for column, title in enumerate(columns):
        axes[0, column].set_title(title, fontsize=9)
    figure.tight_layout()
    return figure


def plot_per_image_scores(per_image: dict[str, Any]) -> plt.Figure:
    """Plot every image's fill fraction, sorted, split above and below the cutoff.

    Args:
        per_image (dict): Output of :func:`analysis.pipeline.run_per_image`.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    order = per_image["order"]
    positions = np.arange(len(order))
    figure, axis = plt.subplots(figsize=(9.0, 4.2))
    axis.plot(positions, per_image["fill_overall"][order], "o-", color="C0", ms=4, label="overall")
    axis.plot(positions, per_image["fill_below"][order], "s-", color="C2", ms=3,
              label=f"below {per_image['cutoff']:.3f} cyc/px")
    axis.plot(positions, per_image["fill_above"][order], "^-", color="C3", ms=3,
              label=f"above {per_image['cutoff']:.3f} cyc/px")
    axis.set_xticks(positions)
    axis.set_xticklabels([str(per_image["labels"][i]) for i in order], rotation=90, fontsize=5.5)
    axis.set_ylabel("fill fraction")
    axis.set_title("Per-image fill fraction, sorted from most determined to most filled in")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def plot_loss_traces(bank) -> plt.Figure:
    """Plot the guidance loss trajectory, to confirm the sampler actually converged.

    A reconstruction whose rate mismatch never falls has not been guided toward
    the latent at all, and its coherence would be measuring the prior alone.

    Args:
        bank: The loaded :class:`analysis.pipeline.Bank`.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    losses = bank.losses.reshape(-1, bank.losses.shape[-1])
    steps = np.arange(losses.shape[1])
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.plot(steps, losses.T, color="C0", alpha=0.12, lw=0.8)
    axis.plot(steps, np.median(losses, axis=0), color="k", lw=2, label="median over reconstructions")
    axis.set_yscale("log")
    axis.set_xlabel("reverse-diffusion step")
    axis.set_ylabel("RGC rate-matching loss")
    axis.set_title("Guidance loss trace")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure
