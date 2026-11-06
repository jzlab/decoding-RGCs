# Diffusion Retina Training — Notes & Future Directions

Things to try later, design decisions we deferred, and open questions.

---

## Timestep weighting for random gradient steps

Currently the `n_gradient_steps` randomly selected timesteps are weighted
**uniformly** when averaging the rate loss. Better options to explore:

- **SNR weighting**: weight step `t` by `SNR(t) = ᾱ_t / (1 − ᾱ_t)`.
  Steps with higher SNR (low noise, closer to the clean image) produce a
  less noisy `predicted_x0` and likely carry a stronger training signal.
- **1/σ_t weighting**: weight by `1 / sqrt(1 − ᾱ_t)`. Upweights steps
  near the clean end of the trajectory.
- **Uniform on log-SNR**: sample selected indices uniformly on the log-SNR
  scale rather than uniformly on the timestep index, matching the implicit
  weighting used in many diffusion model training objectives.

The weighting scheme is easy to add: after computing `rate_losses`, replace
`torch.stack(rate_losses).mean()` with a weighted sum using per-step
`scheduler.alphas_cumprod` values.

---

## Technically correct MAP objective (option B)

The current implementation (**option A**) computes the rate mismatch at `n`
randomly selected, fully detached diffusion states. Gradient flows only
through the learned encoder — the diffusion trajectory itself is untouched.
This is a quick and memory-efficient approximation.

A more principled approach (**option B**) is closer to the original MAP
posterior-sampling objective: at each selected step, actually *apply* the
guidance correction and measure whether it helped.

For each randomly selected step `t`:
1. Get `x_t` from the no-grad trajectory (detached leaf).
2. Run UNet at `t` with `create_graph=True`.
3. Compute `rate_gradient = autograd.grad(rate_loss_t, x_t, create_graph=True)`.
4. Apply the guidance correction: `x_{t−1}_corrected = ddim_step(x_t) − correction`.
5. Run a **second** UNet forward at `t−1` on `x_{t−1}_corrected`.
6. Compute `rate_loss_{t−1}` at the corrected state (or MSE if `t = 0`).
7. Backprop through steps 6→5→4→3→2 to reach encoder params.

**Tradeoffs vs option A:**
- Requires `create_graph=True` (second-order gradients).
- 2 UNet forward passes per selected step instead of 1 (~2× compute).
- Gradient actually captures the effect of the correction on downstream
  trajectory quality, not just the raw rate mismatch at a noisy state.
- Closer to the intent of the original `reconstruct_with_final_stage_gradients`
  approach, but without the memory cost of chaining across all `T` steps.

---

## Per-step MAP training (supersedes options A and B above)

The `random_step` (option A) and `final_stage` (option B-ish) modes both train
against **rate mismatch**, `mean((f_θ(x_recon) − f_θ(x_target))²)`, with the
gradient flowing through *both* sides. Its global minimum is a cell that
outputs a constant — an encoder invariant to its input. Nothing in that
objective rewards the new cell for carrying information.

`per_step_map` replaces it:

1. Guidance is applied at **every** reverse step, so the trajectory is
   genuinely conditioned on the observed rates and `guidance_scale` affects
   both the samples and the gradient. (In `random_step` no correction was ever
   applied; in `final_stage` with `differentiable_steps=1` the loss was read
   *before* the correction — which is why the η=0.035 and η=0.09402 runs have
   bit-identical `rgc_loss` histories.)
2. At each step the correction is differentiated w.r.t. θ (`create_graph=True`)
   and an optimizer update is taken, so there are `num_inference_steps` updates
   per trajectory instead of one.
3. The supervision is image distance, not rate mismatch. The corrected clean
   estimate is recovered by reusing the step's own epsilon:

       x0_corrected = x0 − correction / sqrt(ᾱ_prev)

   so the loss asks directly whether the correction moved the clean estimate
   toward the target — option B's intent at option A's cost, with no second
   U-Net forward pass. The approximation is that epsilon is held fixed across
   the correction; the exact version re-runs the U-Net at `t−1` on the
   corrected state, at roughly 2× compute.

`x0` is detached in that loss: θ reaches it only through `correction`. The
target measurement `f_θ(x_target)` is recomputed at every update step, since an
optimizer step changes θ and the previous backward already freed its graph.

**Why this is not degenerate.** With RMS normalization the correction always
has magnitude η, so only its *direction* is optimized: the objective is purely
whether the measurement gradient points from `x0` toward the target. A constant
encoder yields a zero gradient and therefore a meaningless direction, so
collapse is not a minimum. It is still a plateau — the fixed types dominate the
combined gradient — which is what `rate_budget_weight` and `diversity_weight`
are now non-zero for.

Timestep weighting (the open item above) is partially addressed by
`step_loss_weighting`: `uniform`, `alpha_prev` (default), or `snr`. Note that
with AdamW the per-step weight is largely normalized away in the parameter
update itself; where it still matters is the balance against the regularizers.

### RF parameterization

The old `kernels` property forced non-negative weights summing to one. With
input in [0, 1] that makes every subunit a weighted pixel average — range
exactly [0, 1], no opponency, no centre-surround. The cell could only ever
report local mean luminance, and the four subunits duly collapsed to pairwise
cosine similarity 0.9987. `kernel_parameterization` now defaults to
`zero_mean_unit_norm`, matching the norm convention the fixed DoG types use in
`sk_retina_encoders._spatiotemporal_filter`. `center_surround` gives an explicit
antagonistic pair; `nonnegative_sum1` reproduces the old runs.

### Resolution

Every spatial length in `params.yaml` is in pixels of a 256-px frame. At 64 px
the learned mosaic collapses to 4 cells with 85×85 RFs — each larger than the
whole image. `--learned-rf-scale 0.25` rescales the learned cell type alone
(85→21 px RF), restoring 105 cells. The fixed types are deliberately left
coarse, which does confound the ablation and should be stated in any writeup.

---

## End-to-end MSE training (`end_to_end_mse`)

`per_step_map` reads its supervision one step early. It recovers the corrected
clean estimate by reusing the step's own epsilon,

    x0_corrected = x0 − correction / sqrt(ᾱ_prev)

which holds epsilon fixed across the correction. That is a one-step lookahead:
it asks whether the correction points the *current* clean estimate toward the
target, not whether the finished image is closer.

`end_to_end_mse` removes that approximation. Guidance is applied at every
reverse step as before, but the last `differentiable_steps` corrections are
chained into a single graph, and the loss is the MSE of the **completed**
reconstruction. One optimizer update per trajectory, so the gradient passes
through several successive corrections and every intervening U-Net forward.

**Tradeoffs vs `per_step_map`:**
- `num_inference_steps`× fewer parameter updates per trajectory, so the
  learning rate wants raising back toward the 1e-4 the single-update runs used;
  `learning_rate: 0.00002` is tuned for 100 updates per trajectory.
- Memory grows with `differentiable_steps` — the whole tail is resident. The
  prefix steps are still first-order (they need d(rate loss)/dx_t for the
  correction) but are detached, so they cost no extra memory.
- `step_loss_weighting` does not apply: there is one loss, at the end, so
  there is no per-timestep weight to choose.
- No `create_graph` on the prefix, but the tail steps use it, so
  `force_math_sdp: true` is still required.

The same anti-degeneracy argument holds: with RMS normalization the correction
has fixed magnitude η, so only its direction is learned, and a constant encoder
gives a zero gradient and hence no direction.

`best.pt` is now selected on `mse` — the completed reconstruction — for both
`per_step_map` and `end_to_end_mse`, rather than on the per-step training
proxy. The legacy modes still select on `total_loss`.

---

## Guidance strength (eta) and normalization: measured, 2026-09-29

The production per-step run used `guidance_scale: 0.035` with `rms`. That is
the worst measured setting of those tried. Two settings beat it on *every*
axis simultaneously.

Measured at fresh init, 5 image batches of 4 from `random_images`, T=100,
**all regularizer weights zeroed**. `kernel cos` is the cross-trajectory cosine
of the accumulated gradient over the 28,900 spatial weights (chance 0.0059);
MSE and rate loss are from the completed reconstruction.

| norm | eta    | kernel cos | MSE     | rate loss |
|------|--------|------------|---------|-----------|
| rms  | 0.001  | 0.2658     | 0.01525 | 1.93e-04  |
| rms  | 0.0033 | 0.1185     | 0.00707 | 2.33e-06  |
| rms  | 0.0133 | 0.1061     | 0.00861 | 3.68e-06  |
| rms  | 0.035  | 0.0666     | 0.02530 | 1.66e-05  |
| rms  | 0.1    | 0.0826     | 0.05548 | 8.20e-05  |
| none | 0.001  | 0.1598     | 0.04586 | 2.98e-04  |
| none | 0.0033 | 0.0968     | 0.02714 | 1.25e-04  |
| none | 0.0133 | 0.2531     | 0.02092 | 4.92e-05  |
| none | 0.035  | 0.1505     | 0.02391 | 2.46e-05  |
| none | 0.1    | 0.1418     | 0.02895 | 1.24e-05  |

**Recommendation: keep `rms`, set `guidance_scale` to 0.0033 (or 0.0133).**
Against the production 0.035, `rms`@0.0033 is 3.6x better MSE, 7.1x better rate
loss, 1.8x better gradient coherence. 0.0033 vs 0.0133 is within error on MSE
(n=5 batches), so treat [0.0033, 0.0133] as the band.

**Do not remove RMS normalization.** `none` never wins on reconstruction or
rate matching at any eta. Above a crossover near eta ~ 0.005 it gives 1.7-2.4x
better kernel coherence, but its best cell (0.0133) is still 3x worse MSE and
21x worse rate loss than `rms`@0.0033. The raw gradient RMS varies **7147x**
across the 100 reverse steps (median 2.18e-08, range 6.0e-10 to 4.3e-06), so
without normalization the correction is concentrated on a few high-gradient
steps and the rates never converge. RMS also underpins the anti-degeneracy
argument above: with it, ||corr|| == eta always, so only direction is learned.

### The earlier eta sweep was confounded — do not reuse its numbers

The first sweep included the smoothness/diversity/rate-budget terms in the
gradient it measured. Those are functions of theta alone, so at fresh init they
are bit-identical across trajectories, while the image-loss gradient reaches
theta only through the correction and so scales with eta. Small eta therefore
drove the measured cosine to 1 for free: eta=0.001 read 0.9987 with the
regularizers in and **0.2658** with them zeroed. The tell was the gradient
norms agreeing to 0.3% across four different image batches.

Anything measuring gradient agreement must zero the regularizers first.

Also note the `ALL` row is misleading throughout: the 6 scalars carry ~95% of
the gradient norm and sit at cos ~ 1.0, so `ALL` reads 0.88-0.98 at every eta
while the spatial kernels are at 0.07-0.27. Read `raw_kernels`, never `ALL`.

### Open: coherence may not be the right selection criterion

Across the 10 cells, kernel cosine vs log(rate loss) gives Spearman rho=+0.358,
p=0.31 — no significant relationship either way at n=10. It is therefore *not*
established that higher cross-batch coherence means a better training signal.
The recommendation above does not depend on resolving this, since `rms`@0.0033
wins all three axes, but any future argument resting on cosine alone should
settle this first.

## Two structural problems that no eta setting fixes

**`output_gamma` is dead.** In `_nonlinearity`, every branch uses
`z = beta * (x - gamma)` except `ppc`, which recomputes from `x`:

    positive = F.softplus(x)
    return positive.pow(alpha) / (beta * positive + 1)

`ON_Smooth_Monostratified` uses `type: ppc` for its output nonlinearity, so
`gamma` has no gradient path. It reads exactly 0.0000 for all 100 epochs of the
per-step run. Separately, the `exp` subunit branch is degenerate:
`alpha*exp(b(x-g)) = [alpha*e^{-bg}]*e^{bx}`, so `alpha` and `gamma` share one
exactly flat direction. Between them the 6 scalars carry **4 effective DOF**.

**The learned cell quiets itself monotonically.** Every live scalar drifts in
one direction for the whole run without reversing, consistent with their
cos ~ 1.0 gradient. Effective subunit log-gain `log_alpha - beta*gamma` falls
from -0.175 (epoch 5) to -0.810 (epoch 100). Meanwhile the kernels, at
cos 0.07 under the production eta, mostly random-walk. Under AdamW the
per-parameter update is normalized, so what separates them is not gradient
magnitude but coherence: cos ~ 1.0 accumulates directed drift, cos ~ 0.07 does
not. `rate_budget_weight` was 1e-6, far too small to oppose this.

This is a plausible co-cause of the non-celltype-like RFs independent of eta.
