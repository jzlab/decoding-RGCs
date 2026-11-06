# Spectral information and prior fill-in in RGC-guided diffusion reconstructions

Per spatial frequency: how much information about the target survives into the reconstruction, and how
much of the reconstruction is invented by the diffusion prior rather than determined by the RGC latent.

Generated from the saved artifacts by `make_report.py`. Every number below is read back out of
`main/summary.json`, `guidance_sweep/guidance_sweep.json` and `tests/finite_k_summary.json`.

---

## 0. Headline findings

1. **Below the lowest mosaic Nyquist the reconstruction is essentially fully determined, and the
   sampler is exactly calibrated there.** Up to 0.05774 cyc/px (ON_Parasol) the mean fill
   fraction is 0.0029 and coherence is at ceiling,
   while `S_YY/S_XX` = 1.001. Different seeds reproduce the same image, and its power
   spectrum matches the target's. This held at every guidance strength in the sweep, a 19x range.

2. **The determined-to-filled crossover sits at 0.09658 cyc/px**, **between the two mosaic limits**: 1.7x the lowest (0.05774 cyc/px, ON_Parasol) but only 0.56x the highest (0.1732 cyc/px, OFF_Midget). Replicated at
   0.08569 cyc/px in the
   full-bandwidth control bank.

3. **The finer midget mosaics are barely exploited.** ON_Midget and OFF_Midget sample three times more
   finely than the parasols and reach Nyquist frequencies of 0.1386 and 0.1732 cyc/px, yet
   coherence has already collapsed to near zero by about 0.15 cyc/px and the fill fraction is
   0.87 above the parasol limit. Whatever the
   midget mosaics encode about these images is not surviving into the reconstruction. This is the most
   actionable result here: the spatial resolution the retina model samples at is not the resolution the
   reconstruction delivers, and the gap is roughly a factor of two in frequency.

4. **Above the parasol limit the sampler is systematically too smooth**, `S_YY/S_XX` =
   0.279 (main bank). The direction replicates in the control, the magnitude does not;
   see the control section for why the magnitude is a property of the target ensemble as much as of the
   sampler.

5. **The repository's default configuration is badly over-guided.** See section 2.

---

## 1. What was computed

`X` is a target image, `Z = encoder(X)` the static RGC latent, and `Y` a reconstruction sampled by a
DDPM prior guided toward `Z`. Because guidance uses only `Z`, `X -> Z -> Y` is a Markov chain, and
drawing `K` seeds at fixed `Z` separates what the latent pins down from what the prior re-draws.

**Preprocessing**, applied identically to `X` and every `Y`: same normalization (raw grayscale in
[0,1], no per-image rescaling), per-image mean subtracted, 2D hann
window with power normalization `U = <w^2> = 0.13953`,
2D FFT, DC excluded.

**Mode counting.** A real `H x W` image has 65536
real degrees of freedom and 3.277e+04
independent complex modes. Summing `-0.5*log2(1-gamma^2)` over real degrees of freedom on the full FFT
plane is the *same sum* as summing `-log2(1-gamma^2)` over independent complex modes on one half-plane,
and the full-plane form needs no even/odd special-casing. Note this corrects the original
specification, which multiplied the half-log form by a complex-mode count and so would have halved
every bit figure.

**Analysis 1**: cross- and auto-spectra averaged over images *and* modes within a bin **before** the
coherence ratio is formed; coherence converted to bits via the Borst & Theunissen (1999)
Gaussian-channel bound; shuffle null from derangements; 500
scene-clustered bootstrap resamples for intervals. Three variants: `X` vs a
single sample, `X` vs the sample mean (naive and split-half debiased), and sample vs sample.

**Analysis 2**: within-`Z` variance across seeds with the `K/(K-1)` correction, binned the same way;
fill fraction `F = V_fill / S_YY`; the law of total variance and the error decomposition verified
numerically; pixel-space variance maps; per-image scores.

**Calibration**: the two spectral consequences of `X` and `Y` being exchangeable given `Z`.

---

## 2. Configuration actually used

| setting | value |
|---|---|
| prior | `google/ddpm-bedroom-256` |
| guidance scale | **0.0354** (rms-normalized) |
| reverse-diffusion steps | **100** |
| rate noise std | 1.0 |
| cell types | ON_Parasol, OFF_Parasol, ON_Midget, OFF_Midget |
| targets | N=128 (128 independent scenes), 256x256 |
| samples per latent | K=16 |
| base seed | 20260301 |
| target content limit | 0.2871 cyc/px |

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

| arm | RMS log2 power ratio | median S_YY/S_XX | pixel corr. | MSE(X, Ybar) | bits/image | crossover (cyc/px) |
|---|---|---|---|---|---|---|
| g=0.0133, T=250 | 1.89 | 0.241 | 0.947 | 0.00467 | 6697 | 0.111 |
| g=0.0354, T=100 **(chosen)** | 1.4 | 0.332 | 0.937 | 0.00546 | 4294 | 0.0942 |
| g=0.0354, T=250 | 1.78 | 3.12 | 0.909 | 0.008 | 5321 | 0.0903 |
| g=0.094, T=250 | 3.14 | 8.68 | 0.836 | 0.016 | 3543 | 0.0831 |
| g=0.25, T=250 | 3.25 | 9.22 | 0.827 | 0.0172 | 2794 | 0.0848 |

Criterion: RMS of log2(S_YY/S_XX) over bins below the target content limit; log2 makes over- and under-expression symmetric.

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

| mosaic | lattice spacing (px) | Nyquist min (cyc/px) | Nyquist max (cyc/px) |
|---|---|---|---|
| ON_Parasol | 10 | 0.05774 | 0.06667 |
| OFF_Parasol | 8.333 | 0.06928 | 0.08 |
| ON_Midget | 4.167 | 0.1386 | 0.16 |
| OFF_Midget | 3.333 | 0.1732 | 0.2 |

The four mosaics differ in spacing by a factor of three, so every frequency figure carries four
markers. They are also interleaved, so the union samples more densely than any one of them: the finest
single-mosaic line is a *lower bound* on the union's limit.

---

## 5. Results

### Information

| quantity | value |
|---|---|
| `X` vs single sample `Y_1` | 4980 bits/image (0.076 bits/pixel) |
| same, shuffle-corrected | 4904.7 bits/image |
| `X` vs sample mean (naive) | 7132.9 bits/image |
| `X` vs sample mean (split-half debiased) | 8063.7 bits/image |

All of these are **lower bounds** on `I(X;Y)`.

### Fill-in and the crossover

| quantity | value |
|---|---|
| crossover frequency (`F = 0.5`) | **0.09658 cyc/px** |
| mean fill fraction below lowest Nyquist | 0.00286 |
| mean fill fraction above lowest Nyquist | 0.87 |
| per-image fill fraction | 0.0848 +/- 0.0874 |
| per-image fill below / above cutoff | 0.00487 / 0.476 |

The per-image "above cutoff" score is a power-weighted sum over all modes above the cutoff, so it is
dominated by the frequencies just above it where the spectrum still has most of its power. That is why
it is lower than the bin-averaged fill fraction above the same cutoff, which weights every bin equally.
Use the per-image scores to rank images, and the binned curves to read frequency dependence.

The determined-to-filled crossover sits **between the two mosaic limits**: 1.7x the lowest (0.05774 cyc/px, ON_Parasol) but only 0.56x the highest (0.1732 cyc/px, OFF_Midget).

### Identity checks

Both variance identities close to numerical precision relative to the power in the bin, which
validates the finite-`K` corrections on real data rather than only on the synthetic toy:

| identity | max relative residual |
|---|---|
| law of total variance | 1.1e-15 |
| error decomposition | 5.6e-15 |

The fill-in share of squared error is 0.242 against the
exact-posterior value of 0.469; a calibrated sampler would be flat at
the latter at every frequency.

### Calibration

| diagnostic | value |
|---|---|
| median `S_YY/S_XX` below lowest Nyquist | 1.001 (calibrated) |
| median `S_YY/S_XX` above lowest Nyquist | 0.279 (under-expresses power) |
| median `|S_XY|/S_YjYk` over 18 reliable bins | 1.73 |

S_YY/S_XX below one means the sampler under-expresses power at that frequency; above one means it injects more than the prior would. |S_XY|/S_YjYk below one means two samples agree with each other more than either agrees with the truth, the signature of a shared bias. The power ratio is blind to a wrong posterior mean and the cross ratio is blind to a wrong posterior spread, so both must be read.

### Full-bandwidth control

Main-bank targets are resampled up from small sources. The typical source gives a content limit of
0.4609 cycles/pixel, but the smallest
source in the bank limits the most conservative usable band to
0.2871 cycles/pixel, which is the line drawn on the
figures. A second bank of
50 native-resolution tiles cut from two high-resolution photographs
(full bandwidth to 0.5 cyc/px, K=8) was run at the same operating point
as a control.

| quantity | main bank (resampled) | control bank (native) |
|---|---|---|
| crossover (cyc/px) | 0.09658 | 0.08569 |
| fill fraction below lowest Nyquist | 0.00286 | 0.00443 |
| fill fraction above lowest Nyquist | 0.87 | 0.935 |
| median S_YY/S_XX below lowest Nyquist | 1.001 | 1.005 |
| median S_YY/S_XX above lowest Nyquist | 0.279 | 0.771 |
| median \|S_XY\|/S_YjYk | 1.73 | 1.06 |

**What replicates and what does not.** The crossover, the near-zero fill fraction below the lowest
mosaic Nyquist, the near-total fill fraction above it, and the exact calibration below that Nyquist all
agree closely between the two banks, and the crossover sits well below the content limit in both. Those
conclusions do not depend on the target resampling.

The two calibration numbers *above* the lowest Nyquist do not replicate as cleanly. The power deficit
is much milder in the control (0.771
against 0.279) and the cross ratio is essentially one
(1.06 against
1.73). Both are consistent with the control's
targets simply carrying less high-frequency power: these are *ratios* to `S_XX`, so a smoother target
set moves them toward one without the sampler behaving any differently. The direction of the effect is
the same in both banks, but its magnitude is a property of the target ensemble as much as of the
sampler, and should not be quoted as a sampler constant.

The control's tiles come from only 2 source photographs and use
K=8, so its intervals are wider; it is a bandwidth control, not an
independent replication.

### Per-image extremes

Most determined (lowest fill fraction):

- `scandinavian/bedroom_scandinavian_65` 0.0157
- `modern/bedroom_modern_139` 0.0171
- `minimalist/bedroom_minimalist_21` 0.0183
- `boho/bedroom_boho_133` 0.0206
- `industrial/bedroom_industrial_194` 0.0238

Most filled in:

- `modern/bedroom_modern_127` 0.574
- `scandinavian/bedroom_scandinavian_34` 0.473
- `industrial/bedroom_industrial_179` 0.427
- `modern/bedroom_modern_28` 0.382
- `minimalist/bedroom_minimalist_159` 0.377

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

| K | fill-fraction SE (relative to K=32) | naive mean-coherence max rel. bias | split-half max abs. error |
|---|---|---|---|
| 2 | 3.9 | 0.747 | not defined |
| 3 | 2.51 | 0.68 | not defined |
| 4 | 2.04 | 0.62 | 0.0088 |
| 6 | 1.6 | 0.507 | 0.00864 |
| 8 | 1.41 | 0.446 | 0.00736 |
| 12 | 1.23 | 0.337 | 0.00529 |
| 16 | 1.15 | 0.28 | 0.0046 |
| 24 | 1.05 | 0.217 | 0.00418 |
| 32 | 1 | 0.168 | 0.00317 |

Smallest K within 20 percent of the K=32 fill-fraction standard error, floored at 4 so the split-half debiased estimator is available.

**The naive X-versus-Ybar coherence is still biased low by 17% at K=32 in bins where fill-in dominates. Its bias is (V/K)/(S_mumu + V/K), which no affordable K removes once S_mumu is small. Use the split-half debiased estimator instead; it is flat from K=4 upward.**

K=16 was used for the main run.

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

4. **Windowing.** A hann window suppresses edge leakage at the cost of
   broadening each frequency bin, which smooths sharp spectral features; the power normalization keeps
   totals comparable across window choices but cannot undo the broadening. Bin-to-bin structure finer
   than the window's main lobe should not be interpreted.

5. **Target bandwidth.** Main-bank targets are band-limited to
   0.2871 cyc/px by resampling. Above that, `S_XX` and
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

python diffusion_sample_bank.py --sources <bedroom style dirs> --target-mode resize \
    --n-images 128 --n-samples 16 --base-seed 20260301 \
    --guidance-scale 0.0354 --steps 100 \
    --output results_prior-vs-likelihood/banks/main_n128_k16.npz

python run_spectral_analysis.py \
    --bank results_prior-vs-likelihood/banks/main_n128_k16.npz \
    --output-dir results_prior-vs-likelihood/main

python make_report.py
```

`spectral_fillin_analysis.ipynb` performs the same analysis interactively with the figures inline.
Sampling is checkpointed per image under `banks/<name>.shards/`, so an interrupted run resumes.

No model code was modified. `analysis/spectral_fillin.py` imports nothing but numpy;
`diffusion_sample_bank.py` imports the model code without changing it.

### Artifacts

- `banks/` -- reconstruction banks (`X`, `Y`, seeds, losses, latent rates, config) and run logs
- `main/spectral_fillin_results.npz` -- every curve, interval and per-image score
- `main/summary.json` -- the scalars quoted above
- `main/figures/` -- the figures
- `guidance_sweep/` -- the operating-point selection
- `tests/` -- finite-K sweep and the downsampling overlay check
