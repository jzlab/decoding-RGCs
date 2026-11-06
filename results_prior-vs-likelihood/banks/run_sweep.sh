#!/bin/bash
# Guidance-strength sweep. All arms see identical targets and identical seeds,
# so differences are attributable to the guidance scale alone.
set -e
PY=/home/rotation/anaconda3/envs/lnp_torch/bin/python
BED=/home/rotation/Downloads/archive/interior_design/bedroom
SRC="$BED/boho $BED/industrial $BED/minimalist $BED/modern $BED/scandinavian"
OUT=results_prior-vs-likelihood/banks
export HF_HUB_OFFLINE=1
for g in 0.0133 0.0354 0.0940 0.25; do
  tag=$(echo $g | tr '.' 'p')
  $PY diffusion_sample_bank.py --sources $SRC --target-mode resize \
    --n-images 16 --n-samples 8 --base-seed 20260201 --tile-seed 7 \
    --guidance-scale $g --steps 250 \
    --output $OUT/sweep_g${tag}_T250.npz
done
# One arm at the 100 steps their own search actually used.
$PY diffusion_sample_bank.py --sources $SRC --target-mode resize \
  --n-images 16 --n-samples 8 --base-seed 20260201 --tile-seed 7 \
  --guidance-scale 0.0354 --steps 100 \
  --output $OUT/sweep_g0p0354_T100.npz
echo "SWEEP COMPLETE"
