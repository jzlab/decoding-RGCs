#!/usr/bin/env bash
# Sweep the parameterized ON_Smooth_Monostratified mosaic (params.yaml) added to every
# cell-type combination already present under results/, matching the existing sweep grid
# (20 images x 10 eta x 20 T, sigma_r=1.0, seed=1234).
set -u

cd "$(dirname "$0")"
PY=/home/rotation/anaconda3/envs/lnp_torch/bin/python
TARGETS=/home/rotation/Downloads/archive/random
RESULTS=../results
SM=ON_Smooth_Monostratified

run() {
  local name="$1"; shift
  if [ -f "$RESULTS/$name/results.csv" ]; then
    echo "[skip] $name already has results.csv"
    return
  fi
  echo "[run ] $name :: $*"
  "$PY" diffusion_hyperparameter_search.py \
    --target-image "$TARGETS" \
    --output-dir "$RESULTS/$name" \
    --params params.yaml \
    --diffusion-params diffusion_params.yaml \
    --include-cell-types "$@" \
    --sigma-r 1.0 \
    --seed 1234 \
    >"$RESULTS/$name.log" 2>&1 \
    && echo "[done] $name" || echo "[FAIL] $name (see $RESULTS/$name.log)"
}

run ONPara_OFFPara_ONMidg_OFFMidg_SM ON_Parasol OFF_Parasol ON_Midget OFF_Midget $SM
run SM              $SM
run ONPara_SM       ON_Parasol $SM
run OFFPara_SM      OFF_Parasol $SM
run ONMidg_SM       ON_Midget $SM
run OFFMidg_SM      OFF_Midget $SM
run ONPara_OFFPara_SM   ON_Parasol OFF_Parasol $SM
run ONPara_ONMidg_SM    ON_Parasol ON_Midget $SM
run ONMidg_OFFMidg_SM   ON_Midget OFF_Midget $SM
run OFFPara_OFFMidg_SM  OFF_Parasol OFF_Midget $SM

echo "ALL SWEEPS FINISHED"
