"""Spectral coherence, information bounds, and prior fill-in for image reconstructions.

This module is deliberately free of any model, dataset, or plotting imports.  It
operates only on numpy arrays (torch tensors are accepted and converted at the
public boundary), so it can be unit-tested against synthetic data with analytic
answers before it is ever pointed at a diffusion reconstruction bank.

The intended data layout is the one produced by the sampling driver:

    X : (N, H, W)      target luminance images
    Y : (N, K, H, W)   K reconstructions per target, each from a different seed

Conventions used throughout
---------------------------
* Spatial frequency is in **cycles per pixel**.  ``numpy.fft.fftfreq`` gives
  exactly this when the sample spacing is one pixel.
* Spectra are accumulated over the **full** FFT plane, not an explicit half
  plane.  See :func:`make_setup` and :func:`information_bits` for why this is
  both correct and simpler than half-plane bookkeeping.
* Every per-bin spectrum is a weighted mean over images *and* modes.  Ratios
  such as coherence are formed only **after** that averaging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "as_numpy",
    "make_window",
    "SpectralSetup",
    "make_setup",
    "preprocess",
    "Binning",
    "radial_binning",
    "radial_orientation_binning",
    "BinnedSpectra",
    "per_image_bin_sums",
    "reduce_bin_sums",
    "coherence",
    "information_bits",
    "derangement",
    "shuffle_null_coherence",
    "bootstrap_ci",
    "bin_mean_of_modes",
    "fill_variance_per_mode",
    "fill_curves",
    "total_variance_check",
    "error_decomposition",
    "pixel_variance_maps",
    "per_image_fill_scores",
    "square_nyquist",
    "hex_nyquist",
    "hex_nyquist_bounds",
]


# ---------------------------------------------------------------------------
# Array boundary
# ---------------------------------------------------------------------------


def as_numpy(array: Any, dtype: Any = np.float64) -> np.ndarray:
    """Convert a torch tensor, numpy array, or array-like to a numpy array.

    Args:
        array (Any): Input array, numpy array, or torch tensor.
        dtype (Any): Target numpy dtype, or ``None`` to preserve the input dtype.

    Returns:
        numpy.ndarray: Contiguous numpy array on the host.
    """
    if hasattr(array, "detach"):  # torch tensor, without importing torch
        array = array.detach().cpu().numpy()
    out = np.asarray(array)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def _hann_1d(n: int) -> np.ndarray:
    """Build a periodic-free symmetric Hann window.

    Args:
        n (int): Window length in samples.

    Returns:
        numpy.ndarray: Symmetric Hann window of length ``n``.
    """
    if n == 1:
        return np.ones(1)
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / (n - 1))


def _tukey_1d(n: int, alpha: float) -> np.ndarray:
    """Build a symmetric Tukey (tapered-cosine) window.

    ``alpha = 0`` is a rectangular window and ``alpha = 1`` is a Hann window.

    Args:
        n (int): Window length in samples.
        alpha (float): Fraction of the window spent in the cosine taper.

    Returns:
        numpy.ndarray: Symmetric Tukey window of length ``n``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Tukey alpha must lie in [0, 1].")
    if n == 1:
        return np.ones(1)
    if alpha <= 0.0:
        return np.ones(n)
    if alpha >= 1.0:
        return _hann_1d(n)
    x = np.linspace(0.0, 1.0, n)
    w = np.ones(n)
    left = x < alpha / 2.0
    right = x > 1.0 - alpha / 2.0
    w[left] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[left] - alpha / 2.0)))
    w[right] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[right] - 1.0 + alpha / 2.0)))
    return w


def make_window(shape: tuple[int, int], kind: str = "hann", alpha: float = 0.5) -> np.ndarray:
    """Build a separable 2D analysis window.

    Args:
        shape (tuple[int, int]): Image shape as ``(height, width)``.
        kind (str): ``"hann"``, ``"tukey"``, or ``"none"`` for no tapering.
        alpha (float): Taper fraction, used only when ``kind`` is ``"tukey"``.

    Returns:
        numpy.ndarray: Window shaped ``(height, width)`` with values in [0, 1].
    """
    height, width = shape
    if kind == "none":
        return np.ones((height, width))
    if kind == "hann":
        wy, wx = _hann_1d(height), _hann_1d(width)
    elif kind == "tukey":
        wy, wx = _tukey_1d(height, alpha), _tukey_1d(width, alpha)
    else:
        raise ValueError(f"Unsupported window kind: {kind}")
    return np.outer(wy, wx)


# ---------------------------------------------------------------------------
# FFT setup and mode bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralSetup:
    """Fixed per-image-size FFT geometry shared by every spectral computation.

    Attributes:
        shape (tuple[int, int]): Image shape ``(height, width)``.
        window (numpy.ndarray): Separable 2D analysis window.
        window_power (float): Window power normalization ``U = mean(window**2)``.
        f_y (numpy.ndarray): Vertical frequency per row, in cycles/pixel.
        f_x (numpy.ndarray): Horizontal frequency per column, in cycles/pixel.
        f_radial (numpy.ndarray): Radial frequency per mode, in cycles/pixel.
        theta (numpy.ndarray): Mode orientation in ``[0, pi)`` radians.
        dof_weight (numpy.ndarray): Real degrees of freedom carried by each mode.
        dc_mask (numpy.ndarray): Boolean mask that is ``True`` only at DC.
    """

    shape: tuple[int, int]
    window: np.ndarray
    window_power: float
    f_y: np.ndarray
    f_x: np.ndarray
    f_radial: np.ndarray
    theta: np.ndarray
    dof_weight: np.ndarray
    dc_mask: np.ndarray

    @property
    def n_pixels(self) -> int:
        """int: Number of image pixels, equal to the number of real degrees of freedom."""
        return int(self.shape[0] * self.shape[1])


def make_setup(shape: tuple[int, int], window: str = "hann", alpha: float = 0.5) -> SpectralSetup:
    """Build the FFT geometry, window, and mode-counting weights for one image size.

    Mode counting.  A real ``H x W`` image has exactly ``H * W`` real degrees of
    freedom.  In the full complex FFT plane, modes that are their own conjugate
    (DC, and the Nyquist row/column entries that exist only for even sizes) are
    purely real and carry one real degree of freedom, while every other mode is
    one of a conjugate pair ``(f, -f)`` that jointly carries two.  Because the
    full plane contains both members of each pair and each self-conjugate mode
    exactly once, assigning **one real degree of freedom to every entry of the
    full plane** reproduces the correct total ``H * W`` for both even and odd
    sizes with no special-casing.  Summing a half-plane quantity over complex
    modes and summing the full plane over real degrees of freedom are then the
    same sum; see :func:`information_bits`.

    Args:
        shape (tuple[int, int]): Image shape as ``(height, width)``.
        window (str): Window kind passed to :func:`make_window`.
        alpha (float): Tukey taper fraction, ignored for other window kinds.

    Returns:
        SpectralSetup: Reusable geometry for images of this shape.
    """
    height, width = int(shape[0]), int(shape[1])
    win = make_window((height, width), window, alpha)
    window_power = float(np.mean(win**2))

    f_y = np.fft.fftfreq(height)
    f_x = np.fft.fftfreq(width)
    fy_grid, fx_grid = np.meshgrid(f_y, f_x, indexing="ij")
    f_radial = np.hypot(fy_grid, fx_grid)

    # Orientation is taken modulo pi: a mode and its conjugate partner describe
    # the same oriented grating, and they must land in the same orientation bin.
    theta = np.mod(np.arctan2(fy_grid, fx_grid), np.pi)

    dof_weight = np.ones((height, width))
    dc_mask = np.zeros((height, width), dtype=bool)
    dc_mask[0, 0] = True

    return SpectralSetup(
        shape=(height, width),
        window=win,
        window_power=window_power,
        f_y=f_y,
        f_x=f_x,
        f_radial=f_radial,
        theta=theta,
        dof_weight=dof_weight,
        dc_mask=dc_mask,
    )


def independent_complex_modes(shape: tuple[int, int], exclude_dc: bool = True) -> float:
    """Count the independent complex Fourier modes of a real image.

    Args:
        shape (tuple[int, int]): Image shape as ``(height, width)``.
        exclude_dc (bool): Whether to drop the DC mode from the count.

    Returns:
        float: Independent complex modes, counting self-conjugate modes as one half.
    """
    height, width = int(shape[0]), int(shape[1])
    total = height * width / 2.0
    return total - 0.5 if exclude_dc else total


def preprocess(images: Any, setup: SpectralSetup) -> np.ndarray:
    """Mean-subtract, window, and Fourier-transform a stack of images.

    The same function must be applied to the targets and to every reconstruction,
    with the same :class:`SpectralSetup`, so that their spectra are comparable.

    The transform is normalized as ``fft2(window * centred) / sqrt(U * H * W)``
    with ``U = mean(window**2)``.  This makes Parseval's identity read
    ``sum_modes |Xhat|**2 == sum_pixels (window * centred)**2 / U``, so the total
    power is independent of the window up to sampling noise.

    Args:
        images (Any): Real images shaped ``(..., height, width)``.
        setup (SpectralSetup): Geometry matching the trailing two axes.

    Returns:
        numpy.ndarray: Complex spectra shaped like ``images``, with DC left in
        place; DC is excluded later by the binning, not here.
    """
    arr = as_numpy(images, np.float64)
    if arr.shape[-2:] != setup.shape:
        raise ValueError(f"Image shape {arr.shape[-2:]} does not match setup shape {setup.shape}.")
    centred = arr - arr.mean(axis=(-2, -1), keepdims=True)
    tapered = centred * setup.window
    height, width = setup.shape
    scale = np.sqrt(setup.window_power * height * width)
    return np.fft.fft2(tapered, axes=(-2, -1)) / scale


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Binning:
    """Assignment of Fourier modes to frequency (and optionally orientation) bins.

    Attributes:
        index (numpy.ndarray): Flat bin index per mode, ``-1`` for excluded modes.
        n_bins (int): Number of bins.
        f_centers (numpy.ndarray): Radial bin centre per bin, in cycles/pixel.
        f_edges (numpy.ndarray): Radial bin edges, in cycles/pixel.
        dof (numpy.ndarray): Real degrees of freedom per bin.
        n_modes (numpy.ndarray): Raw mode count per bin, for diagnostics.
        theta_centers (numpy.ndarray | None): Orientation centre per bin, radians.
        theta_edges (numpy.ndarray | None): Orientation bin edges, radians.
        n_radial (int): Number of radial bins.
        n_orient (int): Number of orientation bins, ``1`` when purely radial.
        n_excluded (int): Modes dropped as DC or beyond ``f_max``.
    """

    index: np.ndarray
    n_bins: int
    f_centers: np.ndarray
    f_edges: np.ndarray
    dof: np.ndarray
    n_modes: np.ndarray
    theta_centers: np.ndarray | None = None
    theta_edges: np.ndarray | None = None
    n_radial: int = 0
    n_orient: int = 1
    n_excluded: int = 0

    @property
    def is_oriented(self) -> bool:
        """bool: Whether this binning resolves orientation as well as radius."""
        return self.n_orient > 1

    def reshape_oriented(self, values: np.ndarray) -> np.ndarray:
        """Reshape a flat per-bin array to ``(n_radial, n_orient)``.

        Args:
            values (numpy.ndarray): Array whose last axis runs over bins.

        Returns:
            numpy.ndarray: Array with the last axis split into radius and orientation.
        """
        return values.reshape(*values.shape[:-1], self.n_radial, self.n_orient)


def _finalize_binning(
    index: np.ndarray,
    setup: SpectralSetup,
    n_bins: int,
    f_edges: np.ndarray,
    f_centers: np.ndarray,
    theta_edges: np.ndarray | None,
    theta_centers: np.ndarray | None,
    n_radial: int,
    n_orient: int,
) -> Binning:
    """Accumulate per-bin degrees of freedom and mode counts for a bin assignment.

    Args:
        index (numpy.ndarray): Flat bin index per mode, ``-1`` where excluded.
        setup (SpectralSetup): Geometry supplying the per-mode degree-of-freedom weights.
        n_bins (int): Number of bins.
        f_edges (numpy.ndarray): Radial bin edges in cycles/pixel.
        f_centers (numpy.ndarray): Radial bin centres in cycles/pixel.
        theta_edges (numpy.ndarray | None): Orientation bin edges in radians.
        theta_centers (numpy.ndarray | None): Orientation bin centres in radians.
        n_radial (int): Number of radial bins.
        n_orient (int): Number of orientation bins.

    Returns:
        Binning: Fully populated binning record.
    """
    flat_index = index.ravel()
    keep = flat_index >= 0
    dof = np.bincount(flat_index[keep], weights=setup.dof_weight.ravel()[keep], minlength=n_bins)
    n_modes = np.bincount(flat_index[keep], minlength=n_bins).astype(np.float64)
    return Binning(
        index=index,
        n_bins=n_bins,
        f_centers=f_centers,
        f_edges=f_edges,
        dof=dof,
        n_modes=n_modes,
        theta_centers=theta_centers,
        theta_edges=theta_edges,
        n_radial=n_radial,
        n_orient=n_orient,
        n_excluded=int((~keep).sum()),
    )


def radial_binning(setup: SpectralSetup, n_bins: int = 32, f_max: float = 0.5) -> Binning:
    """Assign modes to equal-width radial frequency bins, excluding DC.

    Modes beyond ``f_max`` are dropped.  With the default ``f_max = 0.5`` this
    removes the corners of the FFT plane, which reach ``sqrt(2) / 2`` cycles per
    pixel but are sampled anisotropically and would otherwise contaminate the
    highest bins.

    Args:
        setup (SpectralSetup): Geometry for the image size in use.
        n_bins (int): Number of radial bins spanning ``(0, f_max]``.
        f_max (float): Upper radial frequency limit, in cycles/pixel.

    Returns:
        Binning: Radial bin assignment with per-bin degrees of freedom.
    """
    f_edges = np.linspace(0.0, f_max, n_bins + 1)
    f_centers = 0.5 * (f_edges[:-1] + f_edges[1:])
    index = np.digitize(setup.f_radial, f_edges) - 1
    index = np.where(setup.f_radial > f_max, -1, index)
    index = np.where(setup.dc_mask, -1, index)
    index = np.clip(index, -1, n_bins - 1)
    return _finalize_binning(
        index, setup, n_bins, f_edges, f_centers, None, None, n_bins, 1
    )


def radial_orientation_binning(
    setup: SpectralSetup, n_bins: int = 16, n_orient: int = 8, f_max: float = 0.5
) -> Binning:
    """Assign modes to a radius-by-orientation grid of bins, excluding DC.

    Orientation is taken modulo ``pi`` so that a mode and its conjugate partner
    share a bin.  Bin ``0`` is centred on ``theta = 0``, which is a horizontal
    frequency, meaning a vertically oriented grating.

    Args:
        setup (SpectralSetup): Geometry for the image size in use.
        n_bins (int): Number of radial bins spanning ``(0, f_max]``.
        n_orient (int): Number of orientation bins spanning ``[0, pi)``.
        f_max (float): Upper radial frequency limit, in cycles/pixel.

    Returns:
        Binning: Flat bin assignment with ``n_bins * n_orient`` bins, ordered
        radius-major so that :meth:`Binning.reshape_oriented` recovers the grid.
    """
    f_edges = np.linspace(0.0, f_max, n_bins + 1)
    f_centers = 0.5 * (f_edges[:-1] + f_edges[1:])
    radial_index = np.digitize(setup.f_radial, f_edges) - 1
    radial_index = np.clip(radial_index, -1, n_bins - 1)

    # Shift by half a bin so the first bin straddles theta = 0 rather than
    # starting there; orientation bins are then symmetric about their centres.
    half = np.pi / (2.0 * n_orient)
    shifted = np.mod(setup.theta + half, np.pi)
    orient_index = np.minimum((shifted / (np.pi / n_orient)).astype(int), n_orient - 1)
    theta_edges = np.linspace(0.0, np.pi, n_orient + 1) - half
    theta_centers = np.arange(n_orient) * (np.pi / n_orient)

    index = radial_index * n_orient + orient_index
    invalid = (radial_index < 0) | (setup.f_radial > f_max) | setup.dc_mask
    index = np.where(invalid, -1, index)
    return _finalize_binning(
        index,
        setup,
        n_bins * n_orient,
        f_edges,
        f_centers,
        theta_edges,
        theta_centers,
        n_bins,
        n_orient,
    )


# ---------------------------------------------------------------------------
# Cross-spectra, coherence, and information
# ---------------------------------------------------------------------------


@dataclass
class BinnedSpectra:
    """Per-image, per-bin spectral accumulations that bootstrapping can re-weight.

    Storing the weighted sums per image, rather than the already-averaged
    spectra, means a bootstrap resample over images is a cheap re-average of
    these rows instead of a full recomputation of the transforms.

    Attributes:
        s_xx (numpy.ndarray): Per-image weighted sums of ``|Xhat|**2``, ``(N, B)``.
        s_yy (numpy.ndarray): Per-image weighted sums of ``|Yhat|**2``, ``(N, B)``.
        s_xy (numpy.ndarray): Per-image weighted sums of ``Xhat * conj(Yhat)``, ``(N, B)``.
        weight (numpy.ndarray): Total mode weight per bin, ``(B,)``.
        binning (Binning): The binning these sums were accumulated under.
    """

    s_xx: np.ndarray
    s_yy: np.ndarray
    s_xy: np.ndarray
    weight: np.ndarray
    binning: Binning

    @property
    def n_images(self) -> int:
        """int: Number of images contributing to the accumulation."""
        return int(self.s_xx.shape[0])


def _bin_sum(values: np.ndarray, binning: Binning, weights: np.ndarray) -> np.ndarray:
    """Accumulate weighted per-mode values into bins for a stack of images.

    Args:
        values (numpy.ndarray): Per-mode values shaped ``(N, H, W)``, real or complex.
        binning (Binning): Bin assignment.
        weights (numpy.ndarray): Per-mode weights shaped ``(H, W)``.

    Returns:
        numpy.ndarray: Weighted sums shaped ``(N, n_bins)``, matching the input dtype.
    """
    flat_index = binning.index.ravel()
    keep = flat_index >= 0
    idx = flat_index[keep]
    w = weights.ravel()[keep]
    flat_values = values.reshape(values.shape[0], -1)[:, keep] * w
    out = np.zeros((values.shape[0], binning.n_bins), dtype=flat_values.dtype)
    for n in range(values.shape[0]):
        if np.iscomplexobj(flat_values):
            out[n].real = np.bincount(idx, weights=flat_values[n].real, minlength=binning.n_bins)
            out[n].imag = np.bincount(idx, weights=flat_values[n].imag, minlength=binning.n_bins)
        else:
            out[n] = np.bincount(idx, weights=flat_values[n], minlength=binning.n_bins)
    return out


def per_image_bin_sums(x_hat: np.ndarray, y_hat: np.ndarray, binning: Binning,
                       setup: SpectralSetup) -> BinnedSpectra:
    """Accumulate auto- and cross-spectral sums per image and per frequency bin.

    Args:
        x_hat (numpy.ndarray): Target spectra shaped ``(N, H, W)``.
        y_hat (numpy.ndarray): Reconstruction spectra shaped ``(N, H, W)``.
        binning (Binning): Bin assignment to accumulate into.
        setup (SpectralSetup): Geometry supplying per-mode degree-of-freedom weights.

    Returns:
        BinnedSpectra: Per-image weighted sums plus the per-bin weight totals.
    """
    if x_hat.shape != y_hat.shape:
        raise ValueError("Target and reconstruction spectra must have the same shape.")
    w = setup.dof_weight
    s_xx = _bin_sum(np.abs(x_hat) ** 2, binning, w)
    s_yy = _bin_sum(np.abs(y_hat) ** 2, binning, w)
    s_xy = _bin_sum(x_hat * np.conj(y_hat), binning, w)
    weight = _bin_sum(np.ones((1,) + setup.shape), binning, w)[0].real
    return BinnedSpectra(s_xx=s_xx, s_yy=s_yy, s_xy=s_xy, weight=weight, binning=binning)


def reduce_bin_sums(spectra: BinnedSpectra, image_index: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Average per-image bin sums into per-bin spectra, coherence, and information.

    The averaging over images and over modes happens here, **before** the
    coherence ratio is formed.  Averaging per-mode coherences instead would bias
    the result upward, which is why this function never sees a per-mode ratio.

    Args:
        spectra (BinnedSpectra): Per-image accumulations from :func:`per_image_bin_sums`.
        image_index (numpy.ndarray | None): Optional image indices to average over,
            which may repeat; used to form bootstrap resamples.

    Returns:
        dict[str, numpy.ndarray]: Keys ``S_XX``, ``S_YY``, ``S_XY``, ``gamma2``,
        ``bits_per_dof``, and ``bits`` (per bin, already multiplied by the bin's
        degrees of freedom).
    """
    idx = np.arange(spectra.n_images) if image_index is None else np.asarray(image_index)
    weight = np.where(spectra.weight > 0, spectra.weight, np.nan)
    s_xx = spectra.s_xx[idx].mean(axis=0) / weight
    s_yy = spectra.s_yy[idx].mean(axis=0) / weight
    s_xy = spectra.s_xy[idx].mean(axis=0) / weight

    denom = s_xx * s_yy
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma2 = np.abs(s_xy) ** 2 / denom
    gamma2 = np.clip(np.nan_to_num(gamma2, nan=0.0), 0.0, 1.0 - 1e-12)
    bits_per_dof = information_bits(gamma2, np.ones_like(gamma2))
    return {
        "S_XX": s_xx.real,
        "S_YY": s_yy.real,
        "S_XY": s_xy,
        "gamma2": gamma2,
        "bits_per_dof": bits_per_dof,
        "bits": bits_per_dof * spectra.binning.dof,
    }


def coherence(s_xx: np.ndarray, s_yy: np.ndarray, s_xy: np.ndarray) -> np.ndarray:
    """Form magnitude-squared coherence from already bin-averaged spectra.

    Args:
        s_xx (numpy.ndarray): Bin-averaged target auto-spectrum.
        s_yy (numpy.ndarray): Bin-averaged reconstruction auto-spectrum.
        s_xy (numpy.ndarray): Bin-averaged cross-spectrum, possibly complex.

    Returns:
        numpy.ndarray: Coherence in ``[0, 1)``, with empty bins mapped to zero.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma2 = np.abs(s_xy) ** 2 / (np.asarray(s_xx) * np.asarray(s_yy))
    return np.clip(np.nan_to_num(gamma2, nan=0.0), 0.0, 1.0 - 1e-12)


def information_bits(gamma2: np.ndarray, dof: np.ndarray) -> np.ndarray:
    """Convert coherence to a Gaussian-channel information rate in bits.

    This is the Borst and Theunissen (1999) coherence bound,
    ``I = -0.5 * log2(1 - gamma^2)`` bits per **real degree of freedom**.  Under
    a jointly Gaussian assumption on the target and the reconstruction it is a
    **lower bound** on the true mutual information ``I(X; Y)``: any non-Gaussian
    structure or non-linear dependence between X and Y carries information that
    coherence cannot see.  It is a bound in that direction only, and it is not a
    bound at all if the Gaussian assumption is used to argue the reverse.

    The per-degree-of-freedom convention matters.  A real image has ``H * W``
    real degrees of freedom and ``H * W / 2`` independent complex modes, and a
    complex mode carries two real degrees of freedom.  Summing
    ``-0.5 * log2(1 - gamma^2)`` over real degrees of freedom is therefore the
    same total as summing ``-log2(1 - gamma^2)`` over independent complex modes.
    Multiplying the half-log form by a *complex* mode count instead would halve
    the answer; :class:`Binning` accumulates real degrees of freedom precisely so
    that this function receives the right multiplier.

    Args:
        gamma2 (numpy.ndarray): Coherence per bin, in ``[0, 1)``.
        dof (numpy.ndarray): Real degrees of freedom per bin.

    Returns:
        numpy.ndarray: Information in bits per bin.
    """
    g2 = np.clip(np.asarray(gamma2, dtype=np.float64), 0.0, 1.0 - 1e-12)
    return -0.5 * np.log2(1.0 - g2) * np.asarray(dof, dtype=np.float64)


# ---------------------------------------------------------------------------
# Bias control: shuffle null and bootstrap
# ---------------------------------------------------------------------------


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a permutation of ``0..n-1`` with no fixed point.

    Args:
        n (int): Number of elements, which must be at least two.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: Index array in which no element maps to itself.
    """
    if n < 2:
        raise ValueError("A derangement requires at least two elements.")
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def shuffle_null_coherence(
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    binning: Binning,
    setup: SpectralSetup,
    n_shuffles: int = 8,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Estimate the finite-sample coherence floor by pairing mismatched images.

    Coherence is biased upward at finite sample size: with ``N`` images, even
    independent signals give ``E[gamma^2] ~ 1/N`` per bin.  Pairing each target
    with another image's reconstruction preserves the marginal spectra and the
    sample size while destroying the true dependence, so the resulting coherence
    estimates that floor directly.

    Args:
        x_hat (numpy.ndarray): Target spectra shaped ``(N, H, W)``.
        y_hat (numpy.ndarray): Reconstruction spectra shaped ``(N, H, W)``.
        binning (Binning): Bin assignment.
        setup (SpectralSetup): Geometry for the image size in use.
        n_shuffles (int): Number of independent derangements to average over.
        seed (int): Seed for the derangement generator.

    Returns:
        dict[str, numpy.ndarray]: Keys ``gamma2_null`` (mean over derangements),
        ``gamma2_null_std``, and ``bits_null``.
    """
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_shuffles):
        perm = derangement(x_hat.shape[0], rng)
        spectra = per_image_bin_sums(x_hat, y_hat[perm], binning, setup)
        nulls.append(reduce_bin_sums(spectra)["gamma2"])
    nulls = np.stack(nulls, axis=0)
    gamma2_null = nulls.mean(axis=0)
    return {
        "gamma2_null": gamma2_null,
        "gamma2_null_std": nulls.std(axis=0, ddof=1) if n_shuffles > 1 else np.zeros_like(gamma2_null),
        "bits_null": information_bits(gamma2_null, binning.dof),
    }


def correct_coherence(gamma2_raw: np.ndarray, gamma2_null: np.ndarray) -> np.ndarray:
    """Subtract the shuffle-null coherence floor from a raw coherence estimate.

    The correction is additive in coherence and clipped at zero, which is the
    usual convention.  It is approximate: the bias is only exactly additive when
    the true coherence is small, so the corrected curve should always be reported
    next to the raw curve and the null, never alone.

    Args:
        gamma2_raw (numpy.ndarray): Raw coherence per bin.
        gamma2_null (numpy.ndarray): Shuffle-null coherence per bin.

    Returns:
        numpy.ndarray: Corrected coherence, clipped to ``[0, 1)``.
    """
    return np.clip(np.asarray(gamma2_raw) - np.asarray(gamma2_null), 0.0, 1.0 - 1e-12)


def bootstrap_ci(
    statistic: Callable[[np.ndarray], np.ndarray],
    n_images: int,
    n_boot: int = 500,
    seed: int = 0,
    percentiles: tuple[float, float] = (2.5, 97.5),
    groups: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Bootstrap a per-bin statistic by resampling images with replacement.

    When the images are not independent, for example when they are tiles cut from
    a handful of source photographs, resampling individual images understates the
    uncertainty: two tiles from one scene carry much of the same information, so
    the effective sample size is closer to the number of scenes than to the number
    of tiles.  Passing ``groups`` switches to a cluster bootstrap that resamples
    whole groups, which is the honest interval in that case.

    Args:
        statistic (Callable): Maps an array of image indices to a per-bin array.
        n_images (int): Number of images available to resample.
        n_boot (int): Number of bootstrap resamples.
        seed (int): Seed for the resampling generator.
        percentiles (tuple[float, float]): Lower and upper percentiles for the interval.
        groups (numpy.ndarray | None): Optional group label per image.  When given,
            whole groups are resampled with replacement instead of single images.

    Returns:
        dict[str, numpy.ndarray]: Keys ``point`` (the statistic on the full
        sample), ``lo``, ``hi``, ``mean``, and ``std`` across resamples.
    """
    rng = np.random.default_rng(seed)
    point = np.asarray(statistic(np.arange(n_images)))
    draws = np.empty((n_boot,) + point.shape, dtype=point.dtype)

    if groups is None:
        members = None
        n_groups = 0
    else:
        groups = np.asarray(groups)
        if groups.shape[0] != n_images:
            raise ValueError("groups must supply one label per image.")
        unique = np.unique(groups)
        members = [np.flatnonzero(groups == label) for label in unique]
        n_groups = len(members)
        if n_groups < 2:
            raise ValueError("A cluster bootstrap needs at least two groups.")

    for b in range(n_boot):
        if members is None:
            index = rng.integers(0, n_images, size=n_images)
        else:
            picked = rng.integers(0, n_groups, size=n_groups)
            index = np.concatenate([members[i] for i in picked])
        draws[b] = statistic(index)
    lo, hi = np.percentile(draws.real, percentiles, axis=0)
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "mean": draws.mean(axis=0),
        "std": draws.std(axis=0, ddof=1),
    }


# ---------------------------------------------------------------------------
# Fill-in: within-Z variance across samples
# ---------------------------------------------------------------------------


def bin_average(values: np.ndarray, binning: Binning, setup: SpectralSetup) -> np.ndarray:
    """Average a per-mode quantity within each bin, separately for each image.

    Args:
        values (numpy.ndarray): Per-mode values shaped ``(N, H, W)``.
        binning (Binning): Bin assignment.
        setup (SpectralSetup): Geometry supplying per-mode degree-of-freedom weights.

    Returns:
        numpy.ndarray: Per-image bin means shaped ``(N, n_bins)``.
    """
    sums = _bin_sum(values, binning, setup.dof_weight)
    weight = _bin_sum(np.ones((1,) + setup.shape), binning, setup.dof_weight)[0].real
    weight = np.where(weight > 0, weight, np.nan)
    return sums / weight


def bin_mean_of_modes(values: np.ndarray, binning: Binning, setup: SpectralSetup) -> np.ndarray:
    """Average a single per-mode map within each bin, with degree-of-freedom weights.

    This is the companion to :func:`bin_average` for quantities that are not
    per-image, such as an analytic power spectrum or a transfer function.  It is
    the right way to turn a theoretical per-mode prediction into a per-bin
    prediction comparable with an estimated curve.

    Note that binning does not commute with non-linear functions.  The bin
    average of ``1 / (1 + SNR(f))`` is not ``1 / (1 + mean SNR)``; to predict a
    binned fill fraction, bin the numerator and denominator separately and divide
    afterwards, exactly as the estimator does.

    Args:
        values (numpy.ndarray): Per-mode map shaped ``(H, W)``.
        binning (Binning): Bin assignment.
        setup (SpectralSetup): Geometry supplying per-mode degree-of-freedom weights.

    Returns:
        numpy.ndarray: Bin means shaped ``(n_bins,)``.
    """
    flat_index = binning.index.ravel()
    keep = flat_index >= 0
    weights = setup.dof_weight.ravel()[keep] * np.asarray(values, dtype=np.float64).ravel()[keep]
    sums = np.bincount(flat_index[keep], weights=weights, minlength=binning.n_bins)
    return sums / np.where(binning.dof > 0, binning.dof, np.nan)


def fill_variance_per_mode(y_hat: np.ndarray) -> np.ndarray:
    """Compute the unbiased within-Z sample variance across reconstruction samples.

    For each image and each Fourier mode this is
    ``V_fill = (1 / (K - 1)) * sum_k |Yhat_k - Ybar|^2``, the complex sample
    variance with Bessel's correction.  It estimates ``Var(Yhat | Z)``: the part
    of the reconstruction that the latent does not pin down and the prior
    supplies anew on every seed.

    Args:
        y_hat (numpy.ndarray): Reconstruction spectra shaped ``(N, K, H, W)``.

    Returns:
        numpy.ndarray: Within-Z variance per image and mode, shaped ``(N, H, W)``.
    """
    if y_hat.ndim != 4:
        raise ValueError("Reconstruction spectra must be shaped (N, K, H, W).")
    k = y_hat.shape[1]
    if k < 2:
        raise ValueError("At least two samples per latent are needed to estimate fill-in.")
    y_bar = y_hat.mean(axis=1, keepdims=True)
    return (np.abs(y_hat - y_bar) ** 2).sum(axis=1) / (k - 1)


@dataclass
class FillAccumulation:
    """Per-image, per-bin quantities needed by every fill-in and identity curve.

    Attributes:
        v_fill (numpy.ndarray): Within-Z variance, ``(N, B)``.
        s_yy (numpy.ndarray): Mean over samples of ``|Yhat|^2``, ``(N, B)``.
        s_xx (numpy.ndarray): Target auto-spectrum, ``(N, B)``.
        s_ybar (numpy.ndarray): ``|Ybar|^2`` for the finite-K sample mean, ``(N, B)``.
        err_total (numpy.ndarray): Mean over samples of ``|Xhat - Yhat_k|^2``, ``(N, B)``.
        err_mean (numpy.ndarray): ``|Xhat - Ybar|^2``, ``(N, B)``.
        n_samples (int): Number of reconstruction samples per latent, K.
        binning (Binning): The binning these quantities were accumulated under.
    """

    v_fill: np.ndarray
    s_yy: np.ndarray
    s_xx: np.ndarray
    s_ybar: np.ndarray
    err_total: np.ndarray
    err_mean: np.ndarray
    n_samples: int
    binning: Binning

    @property
    def n_images(self) -> int:
        """int: Number of images contributing to the accumulation."""
        return int(self.v_fill.shape[0])


def accumulate_fill(
    x_hat: np.ndarray, y_hat: np.ndarray, binning: Binning, setup: SpectralSetup
) -> FillAccumulation:
    """Bin every per-mode quantity the fill-in analysis needs, keeping images separate.

    Args:
        x_hat (numpy.ndarray): Target spectra shaped ``(N, H, W)``.
        y_hat (numpy.ndarray): Reconstruction spectra shaped ``(N, K, H, W)``.
        binning (Binning): Bin assignment.
        setup (SpectralSetup): Geometry for the image size in use.

    Returns:
        FillAccumulation: Per-image, per-bin accumulations ready for averaging
        or bootstrap resampling over the image axis.
    """
    if y_hat.ndim != 4 or x_hat.ndim != 3:
        raise ValueError("Expected targets shaped (N, H, W) and reconstructions shaped (N, K, H, W).")
    if x_hat.shape[0] != y_hat.shape[0] or x_hat.shape[-2:] != y_hat.shape[-2:]:
        raise ValueError("Target and reconstruction spectra are not aligned.")

    k = y_hat.shape[1]
    y_bar = y_hat.mean(axis=1)
    v_fill = fill_variance_per_mode(y_hat)
    diff = x_hat[:, None, :, :] - y_hat
    return FillAccumulation(
        v_fill=bin_average(v_fill, binning, setup).real,
        s_yy=bin_average((np.abs(y_hat) ** 2).mean(axis=1), binning, setup).real,
        s_xx=bin_average(np.abs(x_hat) ** 2, binning, setup).real,
        s_ybar=bin_average(np.abs(y_bar) ** 2, binning, setup).real,
        err_total=bin_average((np.abs(diff) ** 2).mean(axis=1), binning, setup).real,
        err_mean=bin_average(np.abs(x_hat - y_bar) ** 2, binning, setup).real,
        n_samples=k,
        binning=binning,
    )


def fill_curves(acc: FillAccumulation, image_index: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Form the fill fraction, its target-normalized version, and determined fraction.

    Args:
        acc (FillAccumulation): Per-image accumulations from :func:`accumulate_fill`.
        image_index (numpy.ndarray | None): Optional image indices to average over,
            which may repeat; used to form bootstrap resamples.

    Returns:
        dict[str, numpy.ndarray]: Keys ``S_XX``, ``S_YY``, ``V_fill``,
        ``fill_fraction`` (``V_fill / S_YY``), ``fill_fraction_vs_xx``
        (``V_fill / S_XX``), and ``determined_fraction`` (``1 - V_fill / S_YY``).
    """
    idx = np.arange(acc.n_images) if image_index is None else np.asarray(image_index)
    s_xx = acc.s_xx[idx].mean(axis=0)
    s_yy = acc.s_yy[idx].mean(axis=0)
    v_fill = acc.v_fill[idx].mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = v_fill / s_yy
        frac_xx = v_fill / s_xx
    return {
        "S_XX": s_xx,
        "S_YY": s_yy,
        "V_fill": v_fill,
        "fill_fraction": frac,
        "fill_fraction_vs_xx": frac_xx,
        "determined_fraction": 1.0 - frac,
    }


def total_variance_check(acc: FillAccumulation, image_index: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Verify the law of total variance per bin, with the finite-K correction.

    The population identity is
    ``S_YY = Var_n(E[Yhat | Z]) + E_n[Var(Yhat | Z)]``, written here with
    uncentred second moments so that it matches the ``S_YY`` used by the
    coherence analysis.  With finite K the sample mean ``Ybar`` carries an extra
    ``V_fill / K`` of its own variance, so the explained term must be
    debiased by exactly that amount; the corrected identity is then exact in
    expectation rather than approximate.

    Args:
        acc (FillAccumulation): Per-image accumulations from :func:`accumulate_fill`.
        image_index (numpy.ndarray | None): Optional image indices to average over.

    Returns:
        dict[str, numpy.ndarray]: Keys ``lhs`` (``S_YY``), ``explained``
        (debiased ``|Ybar|^2`` term), ``explained_raw``, ``unexplained``
        (``V_fill``), ``rhs``, ``residual``, and ``relative_residual``.
    """
    idx = np.arange(acc.n_images) if image_index is None else np.asarray(image_index)
    lhs = acc.s_yy[idx].mean(axis=0)
    v_fill = acc.v_fill[idx].mean(axis=0)
    explained_raw = acc.s_ybar[idx].mean(axis=0)
    explained = explained_raw - v_fill / acc.n_samples
    rhs = explained + v_fill
    residual = lhs - rhs
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = residual / lhs
    return {
        "lhs": lhs,
        "explained": explained,
        "explained_raw": explained_raw,
        "unexplained": v_fill,
        "rhs": rhs,
        "residual": residual,
        "relative_residual": relative,
    }


def error_decomposition(acc: FillAccumulation, image_index: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Split the per-bin reconstruction error into a bias term and a fill-in term.

    The population identity is
    ``E|Xhat - Yhat_k|^2 = E|Xhat - E[Yhat | Z]|^2 + E Var(Yhat | Z)``.  Its exact
    finite-K sample counterpart is
    ``mean_k |Xhat - Yhat_k|^2 = |Xhat - Ybar|^2 + ((K - 1) / K) * V_fill``,
    which holds identically for every image and mode, not just in expectation.
    Because ``E|Xhat - Ybar|^2 = |Xhat - mu|^2 + V_fill / K``, the two finite-K
    terms are together unbiased for the two population terms, and the debiased
    bias term reported here subtracts that ``V_fill / K``.

    Args:
        acc (FillAccumulation): Per-image accumulations from :func:`accumulate_fill`.
        image_index (numpy.ndarray | None): Optional image indices to average over.

    Returns:
        dict[str, numpy.ndarray]: Keys ``lhs``, ``bias_term`` (finite-K
        ``|Xhat - Ybar|^2``), ``bias_term_debiased``, ``fill_term`` (the
        ``((K - 1) / K) * V_fill`` term that closes the identity exactly),
        ``fill_term_population`` (``V_fill``), ``rhs``, ``residual``, and
        ``relative_residual``.
    """
    idx = np.arange(acc.n_images) if image_index is None else np.asarray(image_index)
    k = acc.n_samples
    lhs = acc.err_total[idx].mean(axis=0)
    bias_term = acc.err_mean[idx].mean(axis=0)
    v_fill = acc.v_fill[idx].mean(axis=0)
    fill_term = (k - 1) / k * v_fill
    rhs = bias_term + fill_term
    residual = lhs - rhs
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = residual / lhs
    return {
        "lhs": lhs,
        "bias_term": bias_term,
        "bias_term_debiased": bias_term - v_fill / k,
        "fill_term": fill_term,
        "fill_term_population": v_fill,
        "rhs": rhs,
        "residual": residual,
        "relative_residual": relative,
    }


def per_image_fill_scores(
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    setup: SpectralSetup,
    cutoff: float,
    f_max: float = 0.5,
) -> dict[str, np.ndarray]:
    """Score each image by how much of its reconstruction the prior supplies.

    The score is a power ratio summed over modes rather than a bin average, so
    it answers "what fraction of this reconstruction's variance is re-drawn by
    the prior" directly.  Saving these lets images be sorted by how much the
    prior fills in.

    Args:
        x_hat (numpy.ndarray): Target spectra shaped ``(N, H, W)``.
        y_hat (numpy.ndarray): Reconstruction spectra shaped ``(N, K, H, W)``.
        setup (SpectralSetup): Geometry for the image size in use.
        cutoff (float): Frequency splitting the low and high bands, in cycles/pixel.
        f_max (float): Upper frequency limit; modes beyond it are ignored.

    Returns:
        dict[str, numpy.ndarray]: Keys ``fill_overall``, ``fill_below``,
        ``fill_above``, ``fill_overall_vs_xx``, and ``power_fraction_above``,
        each shaped ``(N,)``, plus the scalar ``cutoff`` used.
    """
    valid = (~setup.dc_mask) & (setup.f_radial <= f_max)
    low = valid & (setup.f_radial <= cutoff)
    high = valid & (setup.f_radial > cutoff)

    v_fill = fill_variance_per_mode(y_hat)
    s_yy = (np.abs(y_hat) ** 2).mean(axis=1)
    s_xx = np.abs(x_hat) ** 2
    w = setup.dof_weight

    def band_ratio(num: np.ndarray, den: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Sum two per-mode quantities over a frequency band and take their ratio."""
        wm = (w * mask)[None, :, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (num * wm).sum(axis=(-2, -1)) / (den * wm).sum(axis=(-2, -1))

    return {
        "fill_overall": band_ratio(v_fill, s_yy, valid),
        "fill_below": band_ratio(v_fill, s_yy, low),
        "fill_above": band_ratio(v_fill, s_yy, high),
        "fill_overall_vs_xx": band_ratio(v_fill, s_xx, valid),
        "power_fraction_above": band_ratio(s_yy * high[None, :, :], s_yy, valid),
        "cutoff": np.asarray(cutoff),
    }


def pixel_variance_maps(y: Any) -> dict[str, np.ndarray]:
    """Compute per-image and mean within-Z variance maps in pixel space.

    Args:
        y (Any): Reconstructions shaped ``(N, K, H, W)`` in image space.

    Returns:
        dict[str, numpy.ndarray]: Keys ``per_image`` (``(N, H, W)`` unbiased
        variance across samples), ``mean`` (``(H, W)`` average over images),
        ``sample_mean`` (``(N, H, W)`` mean reconstruction per image), and
        ``per_image_scalar`` (``(N,)`` mean variance per image).
    """
    arr = as_numpy(y, np.float64)
    if arr.ndim != 4:
        raise ValueError("Reconstructions must be shaped (N, K, H, W).")
    if arr.shape[1] < 2:
        raise ValueError("At least two samples per latent are needed to estimate fill-in.")
    per_image = arr.var(axis=1, ddof=1)
    return {
        "per_image": per_image,
        "mean": per_image.mean(axis=0),
        "sample_mean": arr.mean(axis=1),
        "per_image_scalar": per_image.mean(axis=(-2, -1)),
    }


# ---------------------------------------------------------------------------
# Sample-vs-sample pairing and the sample mean
# ---------------------------------------------------------------------------


def sample_pair_spectra(y_hat: np.ndarray, binning: Binning, setup: SpectralSetup) -> BinnedSpectra:
    """Build the sample-versus-sample cross-spectrum, averaged over all ordered pairs.

    Averaging ``Yhat_j * conj(Yhat_k)`` over all ``j != k`` collapses to an exact
    closed form, ``S_{Y_j Y_k} = S_YY - V_fill``: the shared part of two
    reconstructions of the same latent is whatever is left of their total power
    once the independently re-drawn part is removed.  Using the closed form
    avoids an ``O(K^2)`` loop and is exact rather than an estimate over sampled
    pairs.  A direct consequence is ``gamma^2_{jk} = (1 - F)^2`` where ``F`` is
    the fill fraction, which the tests check against the looped computation.

    Args:
        y_hat (numpy.ndarray): Reconstruction spectra shaped ``(N, K, H, W)``.
        binning (Binning): Bin assignment.
        setup (SpectralSetup): Geometry for the image size in use.

    Returns:
        BinnedSpectra: Accumulation whose auto-spectra are both ``S_YY`` and
        whose cross-spectrum is ``S_{Y_j Y_k}``.
    """
    s_auto = (np.abs(y_hat) ** 2).mean(axis=1)
    s_cross = s_auto - fill_variance_per_mode(y_hat)
    w = setup.dof_weight
    auto_sums = _bin_sum(s_auto, binning, w)
    cross_sums = _bin_sum(s_cross, binning, w).astype(np.complex128)
    weight = _bin_sum(np.ones((1,) + setup.shape), binning, w)[0].real
    return BinnedSpectra(s_xx=auto_sums, s_yy=auto_sums, s_xy=cross_sums,
                         weight=weight, binning=binning)


def split_half_mean_spectra(
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    binning: Binning,
    setup: SpectralSetup,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate coherence with the sample mean, with and without the finite-K debias.

    The naive estimate pairs ``X`` with ``Ybar`` over all K samples.  Its
    cross-spectrum ``S_{X Ybar}`` is unbiased for ``S_{X mu}`` because
    ``E[Ybar] = mu``, but its denominator ``S_{Ybar Ybar} = |mu|^2 + V_fill / K``
    is inflated by the sample mean's own noise.  The naive coherence with the
    sample mean is therefore biased **downward**, and approaches its limit only
    as ``1 / K``.

    Splitting the samples into two disjoint halves fixes this: ``YbarA`` and
    ``YbarB`` are conditionally independent given Z, so
    ``E[S_{YbarA YbarB}] = S_{mu mu}`` with no ``V_fill`` term.  Substituting it
    for the denominator gives a finite-K-debiased coherence.

    Args:
        x_hat (numpy.ndarray): Target spectra shaped ``(N, H, W)``.
        y_hat (numpy.ndarray): Reconstruction spectra shaped ``(N, K, H, W)``.
        binning (Binning): Bin assignment.
        setup (SpectralSetup): Geometry for the image size in use.
        seed (int): Seed controlling how samples are split into halves.

    Returns:
        dict[str, Any]: Keys ``naive`` (a :class:`BinnedSpectra` for X versus the
        full sample mean), ``half_a``/``half_b`` (X versus each half mean),
        ``s_mumu`` (per-image debiased ``S_{mu mu}`` sums, ``(N, B)``), and
        ``weight`` (per-bin mode weight).
    """
    k = y_hat.shape[1]
    if k < 4:
        raise ValueError("A split-half estimate needs at least four samples per latent.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(k)
    a, b = perm[: k // 2], perm[k // 2 : 2 * (k // 2)]

    y_bar = y_hat.mean(axis=1)
    y_a, y_b = y_hat[:, a].mean(axis=1), y_hat[:, b].mean(axis=1)
    w = setup.dof_weight
    weight = _bin_sum(np.ones((1,) + setup.shape), binning, w)[0].real
    return {
        "naive": per_image_bin_sums(x_hat, y_bar, binning, setup),
        "half_a": per_image_bin_sums(x_hat, y_a, binning, setup),
        "half_b": per_image_bin_sums(x_hat, y_b, binning, setup),
        "s_mumu": _bin_sum((y_a * np.conj(y_b)).real, binning, w),
        "weight": weight,
        "half_indices": (a, b),
    }


def debiased_mean_coherence(split: dict[str, Any], image_index: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Form the finite-K-debiased coherence between the target and the sample mean.

    Args:
        split (dict[str, Any]): Output of :func:`split_half_mean_spectra`.
        image_index (numpy.ndarray | None): Optional image indices to average over.

    Returns:
        dict[str, numpy.ndarray]: Keys ``gamma2_naive``, ``gamma2_debiased``,
        ``S_mumu``, and the corresponding ``bits_naive`` and ``bits_debiased``.
    """
    naive: BinnedSpectra = split["naive"]
    idx = np.arange(naive.n_images) if image_index is None else np.asarray(image_index)
    weight = np.where(split["weight"] > 0, split["weight"], np.nan)

    reduced = reduce_bin_sums(naive, idx)
    s_xx = reduced["S_XX"]
    s_xy = reduced["S_XY"]
    s_mumu = split["s_mumu"][idx].mean(axis=0) / weight
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma2_debiased = np.abs(s_xy) ** 2 / (s_xx * s_mumu)
    gamma2_debiased = np.clip(np.nan_to_num(gamma2_debiased, nan=0.0), 0.0, 1.0 - 1e-12)
    dof = naive.binning.dof
    return {
        "gamma2_naive": reduced["gamma2"],
        "gamma2_debiased": gamma2_debiased,
        "S_mumu": s_mumu,
        "bits_naive": reduced["bits"],
        "bits_debiased": information_bits(gamma2_debiased, dof),
    }


# ---------------------------------------------------------------------------
# Calibration: exchangeability of X and Y given Z
# ---------------------------------------------------------------------------


def calibration_ratios(
    xy: BinnedSpectra,
    yy: BinnedSpectra,
    image_index: np.ndarray | None = None,
    min_shared: float = 0.05,
) -> dict[str, np.ndarray]:
    """Test the two spectral consequences of exchangeability of X and Y given Z.

    If the sampler were an exact posterior sampler under the true image prior,
    then ``X`` and each ``Y_k`` would be exchangeable given ``Z``.  Two testable
    consequences follow per frequency bin:

    * ``S_XX = S_YY``: the target and a reconstruction are marginally identically
      distributed, so their power spectra agree.  ``S_YY / S_XX < 1`` means the
      sampler is too smooth, under-expressing power at that frequency; ``> 1``
      means it injects more than the prior would.
    * ``S_XY = S_{Y_j Y_k}``: a target and a reconstruction share exactly as much
      as two reconstructions do.  ``|S_XY| / S_{Y_j Y_k} < 1`` means two samples
      agree with each other *more* than either agrees with the truth, the
      signature of a shared bias that the latent did not ask for.

    Both ratios are reported directly and in log2, where zero is perfect
    calibration and the sign reads as a direction immediately.

    The cross ratio degenerates where the latent determines nothing.  Once the
    fill fraction reaches one, ``S_XY`` and ``S_{Y_j Y_k}`` are both consistent
    with zero and their ratio is the ratio of two noise terms, which wanders over
    orders of magnitude without meaning anything.  ``shared_fraction``
    (``S_{Y_j Y_k} / S_YY``, equal to ``1 - F``) measures how much shared
    structure is left to compare, and ``reliable`` marks the bins where enough
    remains for the cross ratio to be read.  Bins outside that mask should be
    greyed out on the plots, never interpreted as miscalibration.

    Args:
        xy (BinnedSpectra): Accumulation for the target against one sample.
        yy (BinnedSpectra): Accumulation from :func:`sample_pair_spectra`.
        image_index (numpy.ndarray | None): Optional image indices to average over.
        min_shared (float): Smallest ``shared_fraction`` at which the cross ratio
            is considered interpretable.

    Returns:
        dict[str, numpy.ndarray]: Keys ``ratio_power`` (``S_YY / S_XX``),
        ``ratio_cross`` (``|S_XY| / S_{Y_j Y_k}``), their ``log2_`` counterparts,
        ``shared_fraction``, ``reliable``, and the four underlying spectra.
    """
    idx = np.arange(xy.n_images) if image_index is None else np.asarray(image_index)
    red_xy = reduce_bin_sums(xy, idx)
    red_yy = reduce_bin_sums(yy, idx)
    s_xx, s_yy = red_xy["S_XX"], red_xy["S_YY"]
    s_xy = np.abs(red_xy["S_XY"])
    s_jk = np.abs(red_yy["S_XY"])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_power = s_yy / s_xx
        ratio_cross = s_xy / s_jk
    with np.errstate(invalid="ignore", divide="ignore"):
        shared = red_yy["S_XY"].real / s_yy
    reliable = np.nan_to_num(shared, nan=0.0) >= min_shared
    return {
        "ratio_power": ratio_power,
        "ratio_cross": np.where(reliable, ratio_cross, np.nan),
        "ratio_cross_raw": ratio_cross,
        "log2_ratio_power": np.log2(np.where(ratio_power > 0, ratio_power, np.nan)),
        "log2_ratio_cross": np.log2(np.where(reliable & (ratio_cross > 0), ratio_cross, np.nan)),
        "shared_fraction": shared,
        "reliable": reliable,
        "S_XX": s_xx,
        "S_YY": s_yy,
        "S_XY": s_xy,
        "S_YjYk": s_jk,
    }


# ---------------------------------------------------------------------------
# Mosaic Nyquist overlays
# ---------------------------------------------------------------------------


def square_nyquist(spacing: float) -> float:
    """Return the Nyquist frequency of a square sampling lattice.

    Args:
        spacing (float): Lattice spacing in pixels.

    Returns:
        float: Nyquist frequency in cycles per pixel, ``1 / (2 * spacing)``.
    """
    if spacing <= 0:
        raise ValueError("Lattice spacing must be positive.")
    return 1.0 / (2.0 * spacing)


def hex_nyquist_bounds(spacing: float) -> tuple[float, float]:
    """Return the smallest and largest Nyquist frequencies of a triangular lattice.

    Derivation.  The mosaic code tiles cells on the lattice generated by
    ``a1 = (s, 0)`` and ``a2 = (s / 2, s * sqrt(3) / 2)``, a triangular lattice
    whose nearest-neighbour spacing is ``s``.  Its reciprocal lattice, in cycles
    per pixel, is itself triangular with nearest-neighbour spacing
    ``g = 2 / (sqrt(3) * s)``.  Aliasing sets in at the boundary of the first
    Brillouin zone, the set of frequencies closer to the origin than to any other
    reciprocal lattice point, which for a triangular lattice is a regular hexagon
    with inradius ``g / 2 = 1 / (sqrt(3) * s)`` and circumradius
    ``g / sqrt(3) = 2 / (3 * s)``.

    So a hexagonal mosaic has no single Nyquist frequency: it resolves
    ``2 / (3 * s)`` cycles/pixel along the directions pointing at the zone
    corners and only ``1 / (sqrt(3) * s)`` along the directions pointing at the
    zone faces, a ratio of ``2 / sqrt(3)``, about 15 percent.  The inradius is
    the conservative scalar to draw on a radially averaged plot, because below it
    no orientation aliases.

    Args:
        spacing (float): Nearest-neighbour cell spacing in pixels.

    Returns:
        tuple[float, float]: ``(inradius, circumradius)`` in cycles per pixel.
    """
    if spacing <= 0:
        raise ValueError("Lattice spacing must be positive.")
    inradius = 1.0 / (np.sqrt(3.0) * spacing)
    return float(inradius), float(2.0 / (3.0 * spacing))


def hex_nyquist(spacing: float, theta: Any, corner_offset: float = 0.0) -> np.ndarray:
    """Evaluate the direction-dependent Nyquist frequency of a triangular lattice.

    The first Brillouin zone is a hexagon, so its boundary at angle ``theta`` is
    ``inradius / cos(phi)`` where ``phi`` is the angle to the nearest face
    normal, wrapped into ``[-pi/6, pi/6]``.  For the lattice built by the mosaic
    code the six shortest reciprocal vectors lie at ``30, 90, 150, ...`` degrees,
    so the face normals sit at those angles and the zone corners, where the
    lattice resolves most, sit at ``0, 60, 120, ...`` degrees.  ``corner_offset``
    rotates that pattern for a lattice in a different orientation.

    Angles follow the same convention as :attr:`SpectralSetup.theta`, namely
    ``arctan2(f_y, f_x)`` with ``y`` increasing downward, which matches the
    ``(x, y)`` convention the mosaic tiling itself uses.

    Args:
        spacing (float): Nearest-neighbour cell spacing in pixels.
        theta (Any): Orientation angles in radians, any shape.
        corner_offset (float): Angle in radians of a zone corner, default ``0``.

    Returns:
        numpy.ndarray: Nyquist frequency in cycles per pixel, shaped like ``theta``.
    """
    inradius, _ = hex_nyquist_bounds(spacing)
    angles = np.asarray(theta, dtype=np.float64) - corner_offset
    sector = np.pi / 3.0
    # Angle to the nearest face normal, which sits half a sector from a corner.
    phi = np.mod(angles, sector) - sector / 2.0
    return inradius / np.cos(phi)


def cycles_per_pixel_to_cycles_per_degree(
    f_cycles_per_pixel: Any, microns_per_pixel: float, microns_per_degree: float
) -> np.ndarray:
    """Convert spatial frequency from cycles per pixel to cycles per degree.

    Args:
        f_cycles_per_pixel (Any): Frequencies in cycles per pixel.
        microns_per_pixel (float): Retinal microns subtended by one image pixel.
        microns_per_degree (float): Retinal microns per degree of visual angle.

    Returns:
        numpy.ndarray: Frequencies in cycles per degree.
    """
    if microns_per_pixel <= 0 or microns_per_degree <= 0:
        raise ValueError("Both micron scales must be positive.")
    return np.asarray(f_cycles_per_pixel, dtype=np.float64) * (microns_per_degree / microns_per_pixel)
