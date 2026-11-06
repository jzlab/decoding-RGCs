#!/bin/bash
# Main run and full-bandwidth control, both at the operating point chosen by the
# guidance sweep: g=0.0354, T=100, which is the optimum the repository's own
# hyperparameter search selected and the best-calibrated arm of the sweep.
set -e
PY=/home/rotation/anaconda3/envs/lnp_torch/bin/python
BED=/home/rotation/Downloads/archive/interior_design/bedroom
SRC="$BED/boho $BED/industrial $BED/minimalist $BED/modern $BED/scandinavian"
OUT=results_prior-vs-likelihood/banks
export HF_HUB_OFFLINE=1

# Main bank: 128 independent bedroom scenes, K=16 from the finite-K test.
$PY diffusion_sample_bank.py --sources $SRC --target-mode resize \
  --n-images 128 --n-samples 16 --base-seed 20260301 --tile-seed 11 \
  --guidance-scale 0.0354 --steps 100 \
  --output $OUT/main_n128_k16.npz
echo "MAIN COMPLETE"

# Control bank: native-resolution tiles, full bandwidth to 0.5 cycles/pixel.
# Checks that nothing in the high-frequency conclusions is an artifact of the
# dataset targets being resampled up from 236 pixels.
$PY diffusion_sample_bank.py \
  --sources hyperparameter_search_bedroom/bedroom.png hyperparameter_search_bedroom2/bedroom2.png \
  --target-mode tile --n-images 50 --n-samples 8 --base-seed 20260401 --tile-seed 3 \
  --guidance-scale 0.0354 --steps 100 \
  --output $OUT/control_tiles_n50_k8.npz
echo "CONTROL COMPLETE"
