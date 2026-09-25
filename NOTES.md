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
