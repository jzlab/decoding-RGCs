#!/usr/bin/env bash
# Cell-type combination sweeps, matching the established grid
# (20 images x 10 eta x 20 T, sigma_r=1.0, seed=1234).
#
# Two stages, answering two separate questions:
#   base : the nine subsets of the four major types   -> what do the four types do?
#   sm   : the parameterised ON_Smooth_Monostratified -> what does adding SM contribute?
#          mosaic alone, and added to each subset
#
# The base stage exists because the original non-SM sweeps were run before this
# driver and their commands were never recorded; re-running them here puts both
# halves of number_of_celltypes_analysis.ipynb under one reproducible script.
#
# Either stage runs against either prior. --backbone dps forwards the DPS
# checkpoint flags to diffusion_hyperparameter_search.py and, by default,
# suffixes every output directory with _dps so the diffusers baselines under
# results/<combination>/ are never overwritten.
#
# --etas narrows the guidance grid. The full ten-point scan is only needed to
# locate the optimum; once it is known, a three-point bracket around it keeps the
# per-combination eta selection in number_of_celltypes_analysis.ipynb meaningful
# (16 of 19 bedroom combinations chose 0.0133, but three chose 0.005) at a third
# of the cost. Omit it to run the full grid.
#
#   ./run_sweeps.sh --stage base --backbone dps --etas "0.005 0.0132957397423624 0.0353553390593274"
#   ./run_sweeps.sh --stage sm   --backbone dps
#   ./run_sweeps.sh --stage all                       # original behaviour
set -u

cd "$(dirname "$0")"
PY=/home/rotation/anaconda3/envs/lnp_torch/bin/python
TARGETS=/home/rotation/Downloads/archive/random
RESULTS=../results
SM=ON_Smooth_Monostratified

STAGE=all
BACKBONE=diffusers
DPS_CHECKPOINT=diffusion/models/imagenet256.pt
DPS_PRESET=imagenet256
SUFFIX=
SUFFIX_SET=0
DRY_RUN=0
ETAS=

usage() {
  sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --stage)          STAGE="$2"; shift 2 ;;
    --backbone)       BACKBONE="$2"; shift 2 ;;
    --dps-checkpoint) DPS_CHECKPOINT="$2"; shift 2 ;;
    --dps-preset)     DPS_PRESET="$2"; shift 2 ;;
    --suffix)         SUFFIX="$2"; SUFFIX_SET=1; shift 2 ;;
    --targets)        TARGETS="$2"; shift 2 ;;
    --etas)           ETAS="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    -h|--help)        usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

case "$STAGE" in base|sm|all) ;; *) echo "--stage must be base, sm or all" >&2; exit 1 ;; esac
case "$BACKBONE" in diffusers|dps) ;; *) echo "--backbone must be diffusers or dps" >&2; exit 1 ;; esac

# Default the suffix from the backbone so a DPS run cannot land on top of the
# diffusers results by omission. An explicit --suffix '' overrides that.
if [ "$SUFFIX_SET" -eq 0 ] && [ "$BACKBONE" = "dps" ]; then
  SUFFIX=_dps
fi

GRID_ARGS=()
if [ -n "$ETAS" ]; then
  # Unquoted on purpose: the value is a space-separated list of floats.
  # shellcheck disable=SC2206
  GRID_ARGS=(--etas $ETAS)
fi

BACKBONE_ARGS=(--backbone "$BACKBONE")
if [ "$BACKBONE" = "dps" ]; then
  if [ ! -f "$DPS_CHECKPOINT" ]; then
    echo "DPS checkpoint not found: $DPS_CHECKPOINT" >&2
    echo "Download it with: gdown --folder 1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh -O diffusion/models/" >&2
    exit 1
  fi
  BACKBONE_ARGS+=(--dps-checkpoint "$DPS_CHECKPOINT" --dps-preset "$DPS_PRESET")
fi

echo "stage=$STAGE backbone=$BACKBONE suffix='${SUFFIX:-<none>}' targets=$TARGETS"
echo "etas=${ETAS:-<full grid>}"
[ "$BACKBONE" = "dps" ] && echo "checkpoint=$DPS_CHECKPOINT preset=$DPS_PRESET"

run() {
  local name="$1$SUFFIX"; shift
  # Glob rather than test results.csv exactly: the original all-four sweep wrote
  # results_out_dist.csv, and skipping on any results*.csv keeps reruns idempotent.
  if compgen -G "$RESULTS/$name/results*.csv" >/dev/null; then
    echo "[skip] $name already has a results CSV"
    return
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry ] $name :: $*"
    return
  fi
  echo "[run ] $name :: $*"
  # The log lives inside the result folder, so a combination's output and the
  # record of how it was produced never get separated.
  mkdir -p "$RESULTS/$name"
  "$PY" diffusion/diffusion_hyperparameter_search.py \
    --target-image "$TARGETS" \
    --output-dir "$RESULTS/$name" \
    --params shared/params.yaml \
    --diffusion-params diffusion/diffusion_params.yaml \
    "${BACKBONE_ARGS[@]}" \
    ${GRID_ARGS[@]+"${GRID_ARGS[@]}"} \
    --include-cell-types "$@" \
    --sigma-r 1.0 \
    --seed 1234 \
    >"$RESULTS/$name/$name.log" 2>&1 \
    && echo "[done] $name" || echo "[FAIL] $name (see $RESULTS/$name/$name.log)"
}

# ---------------------------------------------------------------- stage: base
# The nine subsets of the four major types, ordered as BASE_ORDER in
# number_of_celltypes_analysis.ipynb.
if [ "$STAGE" = "base" ] || [ "$STAGE" = "all" ]; then
  echo "--- stage: base (four major cell types) ---"
  run ONPara_OFFPara_ONMidg_OFFMidg ON_Parasol OFF_Parasol ON_Midget OFF_Midget
  run ONPara          ON_Parasol
  run OFFPara         OFF_Parasol
  run ONMidg          ON_Midget
  run OFFMidg         OFF_Midget
  run ONPara_OFFPara  ON_Parasol OFF_Parasol
  run ONPara_ONMidg   ON_Parasol ON_Midget
  run ONMidg_OFFMidg  ON_Midget OFF_Midget
  run OFFPara_OFFMidg OFF_Parasol OFF_Midget
fi

# ------------------------------------------------------------------ stage: sm
if [ "$STAGE" = "sm" ] || [ "$STAGE" = "all" ]; then
  echo "--- stage: sm (ON_Smooth_Monostratified added) ---"
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
fi

echo "ALL SWEEPS FINISHED (stage=$STAGE backbone=$BACKBONE)"
