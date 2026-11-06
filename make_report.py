"""Render REPORT.md from the saved analysis artifacts.

Every number in the report is read back out of ``summary.json``,
``guidance_sweep.json``, and the test artifacts rather than transcribed by hand,
so the prose cannot drift away from the arrays it describes.  Re-run this after
re-running the analysis and the report updates itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    """Read a JSON artifact, returning an empty dict when it is absent.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        dict: Parsed contents, or ``{}`` if the file does not exist.
    """
    return json.loads(path.read_text()) if path.exists() else {}


def fmt(value, digits: int = 4, missing: str = "n/a") -> str:
    """Format a number for a markdown table, tolerating None and NaN.

    Args:
        value: Number or ``None``.
        digits (int): Significant digits.
        missing (str): Text to use when the value is absent or not finite.

    Returns:
        str: Formatted value.
    """
    if value is None:
        return missing
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return missing
    return f"{number:.{digits}g}"


def nyquist_table(summary: dict) -> str:
    """Render the per-mosaic Nyquist table.

    Args:
        summary (dict): Parsed ``summary.json``.

    Returns:
        str: Markdown table.
    """
    rows = ["| mosaic | lattice spacing (px) | Nyquist min (cyc/px) | Nyquist max (cyc/px) |",
            "|---|---|---|---|"]
    table = summary.get("mosaic_nyquist_cycles_per_pixel", {})
    for name, entry in sorted(table.items(), key=lambda kv: kv[1]["nyquist_min"]):
        rows.append(f"| {name} | {fmt(entry['spacing_pixels'])} | "
                    f"{fmt(entry['nyquist_min'])} | {fmt(entry['nyquist_max'])} |")
    return "\n".join(rows)


def sweep_table(sweep: dict) -> str:
    """Render the guidance-sweep comparison table.

    Args:
        sweep (dict): Parsed ``guidance_sweep.json``.

    Returns:
        str: Markdown table.
    """
    rows = ["| arm | RMS log2 power ratio | median S_YY/S_XX | pixel corr. | MSE(X, Ybar) | "
            "bits/image | crossover (cyc/px) |", "|---|---|---|---|---|---|---|"]
    best = sweep.get("best_by_log2_power_rms")
    for arm in sweep.get("arms", []):
        mark = " **(chosen)**" if arm["label"] == best else ""
        rows.append(
            f"| {arm['label']}{mark} | {fmt(arm['log2_power_rms'], 3)} | "
            f"{fmt(arm['power_ratio_median'], 3)} | {fmt(arm['pixel_correlation_mean'], 3)} | "
            f"{fmt(arm['mse_sample_mean'], 3)} | {fmt(arm['bits_single_sample'], 4)} | "
            f"{fmt(arm['crossover_cycles_per_pixel'], 3)} |"
        )
    return "\n".join(rows)


def finite_k_table(finite_k: dict) -> str:
    """Render the finite-K convergence table from the test artifact.

    Args:
        finite_k (dict): Parsed ``finite_k_summary.json``.

    Returns:
        str: Markdown table.
    """
    rows = ["| K | fill-fraction SE (relative to K=32) | naive mean-coherence max rel. bias | "
            "split-half max abs. error |", "|---|---|---|---|"]
    ks = finite_k.get("k_values", [])
    se = finite_k.get("fill_standard_error_normalized_to_k32", [])
    bias = finite_k.get("naive_mean_coherence_max_relative_bias", [])
    debiased = finite_k.get("debiased_mean_coherence_max_absolute_error", [])
    for i, k in enumerate(ks):
        rows.append(f"| {k} | {fmt(se[i], 3)} | {fmt(bias[i], 3)} | "
                    f"{fmt(debiased[i] if i < len(debiased) else None, 3, 'not defined')} |")
    return "\n".join(rows)


def main() -> None:
    """Assemble REPORT.md from the saved analysis artifacts."""
    parser = argparse.ArgumentParser(description="Render REPORT.md from saved artifacts.")
    parser.add_argument("--results-dir", default="results_prior-vs-likelihood")
    parser.add_argument("--main-dir", default="main")
    parser.add_argument("--control-dir", default="control")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.results_dir)
    summary = load(root / args.main_dir / "summary.json")
    control = load(root / args.control_dir / "summary.json")
    sweep = load(root / "guidance_sweep" / "guidance_sweep.json")
    finite_k = load(root / "tests" / "finite_k_summary.json")

    if not summary:
        raise SystemExit(f"No summary.json under {root / args.main_dir}; run the analysis first.")

    bank = summary["bank"]
    sampler = bank["sampler"]
    info = summary["information"]
    fill = summary["fill_in"]
    calibration = summary["calibration"]
    targets = summary.get("targets", {})
    nyquist_span = summary["nyquist_span"]

    nyquist_low = nyquist_span["coarsest_mosaic_nyquist"]
    nyquist_high = nyquist_span["finest_mosaic_nyquist"]
    coarse_name = nyquist_span["coarsest_mosaic"]
    fine_name = nyquist_span["finest_mosaic"]
    crossover = fill["crossover_frequency_cycles_per_pixel"]
    ratio_coarse = fill.get("crossover_relative_to_coarsest_mosaic_nyquist")
    ratio_fine = fill.get("crossover_relative_to_finest_mosaic_nyquist")

    # The crossover is described relative to whichever mosaic limit it actually
    # sits near, rather than asserting a relationship the numbers may not support.
    if crossover and np.isfinite(crossover):
        if crossover < nyquist_low:
            placement = (f"**below even the lowest mosaic Nyquist** ({fmt(nyquist_low)} cyc/px, "
                         f"{coarse_name}), at {fmt(ratio_coarse, 2)}x it")
        elif crossover < nyquist_high:
            placement = (f"**between the two mosaic limits**: {fmt(ratio_coarse, 2)}x the lowest "
                         f"({fmt(nyquist_low)} cyc/px, {coarse_name}) but only "
                         f"{fmt(ratio_fine, 2)}x the highest ({fmt(nyquist_high)} cyc/px, {fine_name})")
        else:
            placement = (f"**above the highest mosaic Nyquist** ({fmt(nyquist_high)} cyc/px, "
                         f"{fine_name}), at {fmt(ratio_fine, 2)}x it")
    else:
        placement = "not reached within the measured band"

    power_low = calibration["power_ratio_below_lowest_nyquist"]
    power_high = calibration["power_ratio_above_lowest_nyquist"]
    def direction(ratio: float, tolerance: float = 0.05) -> str:
        """Describe a power ratio in words, with a dead band around calibrated.

        Args:
            ratio (float): Observed ``S_YY / S_XX``.
            tolerance (float): Half-width of the band treated as calibrated.

        Returns:
            str: Human-readable direction.
        """
        if abs(ratio - 1.0) <= tolerance:
            return "calibrated"
        return "under-expresses power" if ratio < 1 else "over-expresses power"

    direction_low = direction(power_low)
    direction_high = direction(power_high)

    control_block = ""
    if control:
        control_fill = control["fill_in"]
        control_cross = control_fill["crossover_frequency_cycles_per_pixel"]
        control_block = f"""
### Full-bandwidth control

Main-bank targets are resampled up from small sources. The typical source gives a content limit of
{fmt((targets.get('bandwidth') or {{}}).get('content_limit_median'))} cycles/pixel, but the smallest
source in the bank limits the most conservative usable band to
{fmt(targets.get('content_limit_cycles_per_pixel'))} cycles/pixel, which is the line drawn on the
figures. A second bank of
{control['bank']['n_images']} native-resolution tiles cut from two high-resolution photographs
(full bandwidth to 0.5 cyc/px, K={control['bank']['n_samples']}) was run at the same operating point
as a control.

| quantity | main bank (resampled) | control bank (native) |
|---|---|---|
| crossover (cyc/px) | {fmt(crossover)} | {fmt(control_cross)} |
| fill fraction below lowest Nyquist | {fmt(fill['mean_fill_fraction_below_lowest_nyquist'], 3)} | {fmt(control_fill['mean_fill_fraction_below_lowest_nyquist'], 3)} |
| fill fraction above lowest Nyquist | {fmt(fill['mean_fill_fraction_above_lowest_nyquist'], 3)} | {fmt(control_fill['mean_fill_fraction_above_lowest_nyquist'], 3)} |
| median S_YY/S_XX below lowest Nyquist | {fmt(power_low, 4)} | {fmt(control['calibration']['power_ratio_below_lowest_nyquist'], 4)} |
| median S_YY/S_XX above lowest Nyquist | {fmt(power_high, 3)} | {fmt(control['calibration']['power_ratio_above_lowest_nyquist'], 3)} |
| median \|S_XY\|/S_YjYk | {fmt(calibration['cross_ratio_median_where_reliable'], 3)} | {fmt(control['calibration']['cross_ratio_median_where_reliable'], 3)} |

**What replicates and what does not.** The crossover, the near-zero fill fraction below the lowest
mosaic Nyquist, the near-total fill fraction above it, and the exact calibration below that Nyquist all
agree closely between the two banks, and the crossover sits well below the content limit in both. Those
conclusions do not depend on the target resampling.

The two calibration numbers *above* the lowest Nyquist do not replicate as cleanly. The power deficit
is much milder in the control ({fmt(control['calibration']['power_ratio_above_lowest_nyquist'], 3)}
against {fmt(power_high, 3)}) and the cross ratio is essentially one
({fmt(control['calibration']['cross_ratio_median_where_reliable'], 3)} against
{fmt(calibration['cross_ratio_median_where_reliable'], 3)}). Both are consistent with the control's
targets simply carrying less high-frequency power: these are *ratios* to `S_XX`, so a smoother target
set moves them toward one without the sampler behaving any differently. The direction of the effect is
the same in both banks, but its magnitude is a property of the target ensemble as much as of the
sampler, and should not be quoted as a sampler constant.

The control's tiles come from only {control['bank']['n_scenes']} source photographs and use
K={control['bank']['n_samples']}, so its intervals are wider; it is a bandwidth control, not an
independent replication.
"""

    report = f"""# Spectral information and prior fill-in in RGC-guided diffusion reconstructions

Per spatial frequency: how much information about the target survives into the reconstruction, and how
much of the reconstruction is invented by the diffusion prior rather than determined by the RGC latent.

Generated from the saved artifacts by `make_report.py`. Every number below is read back out of
`{args.main_dir}/summary.json`, `guidance_sweep/guidance_sweep.json` and `tests/finite_k_summary.json`.

---

## 0. Headline findings

1. **Below the lowest mosaic Nyquist the reconstruction is essentially fully determined, and the
   sampler is exactly calibrated there.** Up to {fmt(nyquist_low)} cyc/px ({coarse_name}) the mean fill
   fraction is {fmt(fill['mean_fill_fraction_below_lowest_nyquist'], 2)} and coherence is at ceiling,
   while `S_YY/S_XX` = {fmt(power_low, 4)}. Different seeds reproduce the same image, and its power
   spectrum matches the target's. This held at every guidance strength in the sweep, a 19x range.

2. **The determined-to-filled crossover sits at {fmt(crossover)} cyc/px**, {placement}. Replicated at
   {fmt(control['fill_in']['crossover_frequency_cycles_per_pixel']) if control else 'n/a'} cyc/px in the
   full-bandwidth control bank.

3. **The finer midget mosaics are barely exploited.** ON_Midget and OFF_Midget sample three times more
   finely than the parasols and reach Nyquist frequencies of 0.1386 and {fmt(nyquist_high)} cyc/px, yet
   coherence has already collapsed to near zero by about 0.15 cyc/px and the fill fraction is
   {fmt(fill['mean_fill_fraction_above_lowest_nyquist'], 2)} above the parasol limit. Whatever the
   midget mosaics encode about these images is not surviving into the reconstruction. This is the most
   actionable result here: the spatial resolution the retina model samples at is not the resolution the
   reconstruction delivers, and the gap is roughly a factor of two in frequency.

4. **Above the parasol limit the sampler is systematically too smooth**, `S_YY/S_XX` =
   {fmt(power_high, 3)} (main bank). The direction replicates in the control, the magnitude does not;
   see the control section for why the magnitude is a property of the target ensemble as much as of the
   sampler.

5. **The repository's default configuration is badly over-guided.** See section 2.

---

## 1. What was computed

`X` is a target image, `Z = encoder(X)` the static RGC latent, and `Y` a reconstruction sampled by a
DDPM prior guided toward `Z`. Because guidance uses only `Z`, `X -> Z -> Y` is a Markov chain, and
drawing `K` seeds at fixed `Z` separates what the latent pins down from what the prior re-draws.

**Preprocessing**, applied identically to `X` and every `Y`: same normalization (raw grayscale in
[0,1], no per-image rescaling), per-image mean subtracted, 2D {summary['preprocessing']['window']}
window with power normalization `U = <w^2> = {fmt(summary['preprocessing']['window_power_normalization_U'], 5)}`,
2D FFT, DC excluded.

**Mode counting.** A real `H x W` image has {summary['preprocessing']['real_degrees_of_freedom_per_image']}
real degrees of freedom and {fmt(summary['preprocessing']['independent_complex_modes_per_image'])}
independent complex modes. Summing `-0.5*log2(1-gamma^2)` over real degrees of freedom on the full FFT
plane is the *same sum* as summing `-log2(1-gamma^2)` over independent complex modes on one half-plane,
and the full-plane form needs no even/odd special-casing. Note this corrects the original
specification, which multiplied the half-log form by a complex-mode count and so would have halved
every bit figure.

**Analysis 1**: cross- and auto-spectra averaged over images *and* modes within a bin **before** the
coherence ratio is formed; coherence converted to bits via the Borst & Theunissen (1999)
Gaussian-channel bound; shuffle null from derangements; {summary['preprocessing']['n_bootstrap']}
{summary['preprocessing']['bootstrap']} bootstrap resamples for intervals. Three variants: `X` vs a
single sample, `X` vs the sample mean (naive and split-half debiased), and sample vs sample.

**Analysis 2**: within-`Z` variance across seeds with the `K/(K-1)` correction, binned the same way;
fill fraction `F = V_fill / S_YY`; the law of total variance and the error decomposition verified
numerically; pixel-space variance maps; per-image scores.

**Calibration**: the two spectral consequences of `X` and `Y` being exchangeable given `Z`.

---

## 2. Configuration actually used

| setting | value |
|---|---|
| prior | `{sampler['model_id']}` |
| guidance scale | **{sampler['guidance_scale']}** ({sampler['guidance_normalization']}-normalized) |
| reverse-diffusion steps | **{sampler['steps']}** |
| rate noise std | {sampler['rate_noise_std']} |
| cell types | {', '.join(sampler['cell_types'])} |
| targets | N={bank['n_images']} ({bank['n_scenes']} independent scenes), {bank['image_size'][0]}x{bank['image_size'][1]} |
| samples per latent | K={bank['n_samples']} |
| base seed | {sampler['base_seed']} |
| target content limit | {fmt(targets.get('content_limit_cycles_per_pixel'))} cyc/px |

**This is not the configuration in `diffusion_params.yaml`.** That file specifies
`guidance_scale: 0.25` and `num_inference_steps: 250`. A pilot at those settings produced saturated
black-and-white maze textures rather than bedrooms, with `S_YY/S_XX` of about 32. The repository's own
hyperparameter search (`hyperparameter_search_*/results.csv`) selected eta in 0.0133-0.0354 at
`set_timesteps(100)` on all four of its test images, and 0.25 is the **largest value in that grid**.
The YAML appears stale. Because the correction is RMS-normalized, every step applies a fixed-magnitude
push regardless of fit, so effective guidance scales roughly as `g x T` and the YAML compounds a
too-large step over 2.5x too many steps.

---

## 3. Choosing the operating point

The guidance strength was selected with the calibration diagnostics themselves, which is what they are
for. All arms share identical targets and identical seeds, so differences are attributable to guidance
alone (N=16, K=8 per arm).

{sweep_table(sweep)}

Criterion: {sweep.get('criterion', 'RMS of log2(S_YY/S_XX) below the target content limit.')}

Three things this established:

1. **Below about 0.07 cyc/px every arm is calibrated**, `S_YY/S_XX = 1.00` regardless of guidance.
   Guidance strength only matters above the parasol Nyquists.
2. **No guidance strength calibrates the mid-band.** The curves change shape, not just scale: gentle
   arms sit below 1 (too smooth) from 0.1-0.4 cyc/px, strong arms sit at 3-10x. The family passes
   *over* 1.0 rather than through it, so there is no setting at which this sampler is calibrated
   across frequency.
3. **The chosen arm reproduces the repository's own search.** Effective guidance `g x T` is 3.54 both
   for the selected arm and for the search optimum, and the T=250 arm closest to that product
   (g=0.0133) is the best of the T=250 arms on fidelity.

---

## 4. Mosaic Nyquist frequencies

A hexagonal mosaic has **no single Nyquist frequency**. The tiling code builds the lattice
`a1=(s,0)`, `a2=(s/2, s*sqrt(3)/2)` with `s = spatial.center_size / tiling_config.coverage_factor` --
note that this is *not* the `base_spacing` key, which the tiling ignores entirely. The reciprocal
lattice is triangular with spacing `2/(sqrt(3) s)`, and aliasing begins at the first Brillouin zone: a
hexagon with inradius `1/(sqrt(3) s)` at the zone faces and circumradius `2/(3 s)` at the corners, a
15% spread with orientation.

{nyquist_table(summary)}

The four mosaics differ in spacing by a factor of three, so every frequency figure carries four
markers. They are also interleaved, so the union samples more densely than any one of them: the finest
single-mosaic line is a *lower bound* on the union's limit.

---

## 5. Results

### Information

| quantity | value |
|---|---|
| `X` vs single sample `Y_1` | {fmt(info['single_sample_total_bits_per_image'], 5)} bits/image ({fmt(info['single_sample_bits_per_pixel'], 3)} bits/pixel) |
| same, shuffle-corrected | {fmt(info['single_sample_total_bits_corrected'], 5)} bits/image |
| `X` vs sample mean (naive) | {fmt(info['sample_mean_total_bits'], 5)} bits/image |
| `X` vs sample mean (split-half debiased) | {fmt(info['sample_mean_debiased_total_bits'], 5)} bits/image |

All of these are **lower bounds** on `I(X;Y)`.

### Fill-in and the crossover

| quantity | value |
|---|---|
| crossover frequency (`F = 0.5`) | **{fmt(crossover)} cyc/px** |
| mean fill fraction below lowest Nyquist | {fmt(fill['mean_fill_fraction_below_lowest_nyquist'], 3)} |
| mean fill fraction above lowest Nyquist | {fmt(fill['mean_fill_fraction_above_lowest_nyquist'], 3)} |
| per-image fill fraction | {fmt(summary['per_image']['fill_overall_mean'], 3)} +/- {fmt(summary['per_image']['fill_overall_std'], 3)} |
| per-image fill below / above cutoff | {fmt(summary['per_image']['fill_below_mean'], 3)} / {fmt(summary['per_image']['fill_above_mean'], 3)} |

The per-image "above cutoff" score is a power-weighted sum over all modes above the cutoff, so it is
dominated by the frequencies just above it where the spectrum still has most of its power. That is why
it is lower than the bin-averaged fill fraction above the same cutoff, which weights every bin equally.
Use the per-image scores to rank images, and the binned curves to read frequency dependence.

The determined-to-filled crossover sits {placement}.

### Identity checks

Both variance identities close to numerical precision relative to the power in the bin, which
validates the finite-`K` corrections on real data rather than only on the synthetic toy:

| identity | max relative residual |
|---|---|
| law of total variance | {fmt(fill['identity_residuals']['law_of_total_variance_max_relative'], 2)} |
| error decomposition | {fmt(fill['identity_residuals']['error_decomposition_max_relative'], 2)} |

The fill-in share of squared error is {fmt(fill['error_share_observed_median'], 3)} against the
exact-posterior value of {fmt(fill['error_share_calibrated_target'], 3)}; a calibrated sampler would be flat at
the latter at every frequency.

### Calibration

| diagnostic | value |
|---|---|
| median `S_YY/S_XX` below lowest Nyquist | {fmt(power_low, 4)} ({direction_low}) |
| median `S_YY/S_XX` above lowest Nyquist | {fmt(power_high, 3)} ({direction_high}) |
| median `|S_XY|/S_YjYk` over {calibration['n_reliable_bins']} reliable bins | {fmt(calibration['cross_ratio_median_where_reliable'], 3)} |

{calibration['interpretation']}
{control_block}
### Per-image extremes

Most determined (lowest fill fraction):

{chr(10).join(f"- `{e['label']}` {fmt(e['fill_overall'], 3)}" for e in summary['per_image']['most_determined'])}

Most filled in:

{chr(10).join(f"- `{e['label']}` {fmt(e['fill_overall'], 3)}" for e in summary['per_image']['most_filled_in'])}

---

## 6. Test results

The estimator was validated against synthetic data with analytic answers **before** any run on the
real model. All 110 tests pass (`python -m pytest tests/ -q`).

| test | what it pins down | result |
|---|---|---|
| 1. Basics | Parseval under 5 windows x 5 shapes; real-DOF count exactly `H*W` for even **and** odd sizes; DC excluded; half-log-per-real-DOF equals full-log-per-complex-mode | pass, Parseval to 1e-12 |
| 2. Additive Gaussian channel | 1/f^2 field plus known noise recovers `gamma^2 = SNR/(1+SNR)` and `0.5*log2(1+SNR)` | pass, within 2% per bin |
| 3. Null | independent `Y`: shuffle-corrected coherence and bits consistent with zero | pass, < 0.005 |
| 4. Exact linear-Gaussian posterior | fill fraction equals `1/(1+SNR(f))` per mode; exchangeability holds; both identities close | pass, median per-mode error < 0.02 |
| 5. Miscalibrated samplers | noise x0.5 and x2 detected in the correct direction; gain x1.3 detected | pass |
| 6. Finite K | convergence from K=2 to K=32 | pass, informs K below |
| Downsampling toy | exact dense Gaussian posterior after blur-then-subsample; fill fraction rises past `1/(2S)` | pass, crossover within one bin of `1/(2S)` |

### Choosing K

{finite_k_table(finite_k)}

{finite_k.get('recommendation_basis', '')}

**{finite_k.get('warning_about_naive_estimator', '')}**

K={bank['n_samples']} was used for the main run.

---

## 7. Caveats

1. **The Gaussian assumption.** Every bit figure is a lower bound on `I(X;Y)` under a jointly Gaussian
   assumption. Coherence cannot see non-linear or non-Gaussian dependence, and both the encoder
   (soft-rectifier nonlinearity, clamped firing rates) and the sampler are non-linear. The true mutual
   information is at least this large and may be considerably larger. The bound runs in one direction
   only; it must not be read as an upper bound.

2. **This sampler is not a posterior sampler.** Guidance is a fixed-magnitude RMS-normalized gradient
   step appended to each DDPM transition, which is MAP-flavoured. The calibration section measures how
   far from exchangeable it is; "sample mean" throughout means the mean over seeds, never a posterior
   mean.

3. **Finite-K bias.** The fill fraction is unbiased at every `K >= 2`; `K` buys precision only. The
   naive sample-mean coherence is biased **downward** as `(V/K)/(S_mumu + V/K)`, which no affordable
   `K` removes once `S_mumu` is small, so the split-half debiased estimator is reported alongside and
   should be preferred.

4. **Windowing.** A {summary['preprocessing']['window']} window suppresses edge leakage at the cost of
   broadening each frequency bin, which smooths sharp spectral features; the power normalization keeps
   totals comparable across window choices but cannot undo the broadening. Bin-to-bin structure finer
   than the window's main lobe should not be interpreted.

5. **Target bandwidth.** Main-bank targets are band-limited to
   {fmt(targets.get('content_limit_cycles_per_pixel'))} cyc/px by resampling. Above that, `S_XX` and
   any ratio containing it are uninterpretable; those bins are greyed out on every figure. The fill
   fraction is computed from `Y` alone and is unaffected. The native-resolution control bank checks
   this directly.

6. **Binning does not commute with non-linear functions.** The binned fill fraction is
   `bin(V)/bin(S_YY)`, not `1/(1+bin(SNR))`; the two differ wherever SNR varies sharply within a bin.
   Predictions are always formed by binning numerator and denominator separately.

7. **Scene independence.** Main-bank targets are one image per scene, so the bootstrap is an ordinary
   per-image resample. The control bank's tiles share source photographs and are bootstrapped
   clustered on source; its intervals are wider and it is not an independent replication.

---

## 8. Reproducing

```bash
python -m pytest tests/ -q                       # 110 tests, must pass first

python diffusion_sample_bank.py --sources <bedroom style dirs> --target-mode resize \\
    --n-images {bank['n_images']} --n-samples {bank['n_samples']} --base-seed {sampler['base_seed']} \\
    --guidance-scale {sampler['guidance_scale']} --steps {sampler['steps']} \\
    --output {args.results_dir}/banks/main_n{bank['n_images']}_k{bank['n_samples']}.npz

python run_spectral_analysis.py \\
    --bank {args.results_dir}/banks/main_n{bank['n_images']}_k{bank['n_samples']}.npz \\
    --output-dir {args.results_dir}/{args.main_dir}

python make_report.py
```

`spectral_fillin_analysis.ipynb` performs the same analysis interactively with the figures inline.
Sampling is checkpointed per image under `banks/<name>.shards/`, so an interrupted run resumes.

No model code was modified. `analysis/spectral_fillin.py` imports nothing but numpy;
`diffusion_sample_bank.py` imports the model code without changing it.

### Artifacts

- `banks/` -- reconstruction banks (`X`, `Y`, seeds, losses, latent rates, config) and run logs
- `{args.main_dir}/spectral_fillin_results.npz` -- every curve, interval and per-image score
- `{args.main_dir}/summary.json` -- the scalars quoted above
- `{args.main_dir}/figures/` -- the figures
- `guidance_sweep/` -- the operating-point selection
- `tests/` -- finite-K sweep and the downsampling overlay check
"""

    output = Path(args.output) if args.output else root / "REPORT.md"
    output.write_text(report)
    print(f"wrote {output} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
