#!/bin/bash

cd "$(dirname "$0")" || exit 1
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -z "$REPO_ROOT" || ! -d "$REPO_ROOT/overcooked_v2_experiments" ]]; then
  REPO_ROOT="/home/mlic/mingukang/ex-overcookedv2/experiments-stablock"
fi

# -----------------------------------------------------------------------------
# PH2 Experiment Factory
# -----------------------------------------------------------------------------
EXP="rnn-ph2"
ENV_DEVICE="gpu"
NENVS=64
NUM_SEEDS=5
FIXED_SEED=42
OLD_OVERCOOKED="${OLD_OVERCOOKED:-1}"  # 1이면 old overcooked(v1) 엔진 강제

while [[ $# -gt 0 ]]; do
  case "$1" in
    --old-overcooked) OLD_OVERCOOKED="1"; shift ;;
    --no-old-overcooked) OLD_OVERCOOKED="0"; shift ;;
    *)
      echo "[WARN] Unknown option ignored: $1"
      shift
      ;;
  esac
done

NSTEPS=256
# Shared PH1 blocked-target params
: "${PH1_BETA:=1.0}"
: "${PH1_BETA_SCHEDULE_ENABLED:=True}"
: "${PH1_BETA_START:=0.0}"
: "${PH1_BETA_END:=1.0}"
: "${PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS:=-1}"
: "${PH1_OMEGA:=90.0}"
: "${PH1_SIGMA:=3.0}"
: "${PH1_DIST_THRESH:=0.1}"
: "${PH1_POOL_SIZE:=128}"
: "${PH1_NORMAL_PROB:=0.5}"
: "${PH1_MULTI_PENALTY_ENABLED:=True}"
: "${PH1_MULTI_PENALTY_SINGLE_WEIGHT:=1.0}"
: "${PH1_MULTI_PENALTY_OTHER_WEIGHT:=1.0}"
: "${PH1_EPSILON:=0.2}"
: "${PH2_EPSILON:=0.2}"
: "${PH1_WARMUP_STEPS:=2000000}"
: "${PH1_EVAL_ENABLED:=False}"
: "${PH1_EVAL_EVERY_ENV_STEPS:=10000000}"
: "${PH1_EVAL_VIDEO_EVERY_ENV_STEPS:=10000000}"
: "${PH1_EVAL_DEFER_VIDEO:=True}"
: "${PH1_EVAL_DISABLE_JIT:=True}"
: "${PH1_EVAL_OFFLINE_ONLY:=True}"
: "${PH1_EVAL_LOG_VIDEO:=False}"
: "${PH1_EVAL_VIZ_EPISODES:=0}"
: "${PH1_EVAL_NUM_SEEDS:=1}"
: "${SHARED_PREDICTION:=False}"
: "${ACTION_PREDICTION:=True}"
: "${STATE_PREDICTION:=False}"
: "${PRED_Z_DIM:=64}"
: "${STATE_PRED_LOSS_COEF:=1.0}"

# PH2 schedule configs
PH2_RATIO_STAGE1=2
PH2_RATIO_STAGE2=1
PH2_RATIO_STAGE3=2
PH2_FIXED_IND_PROB=""

run_ph2() {
  local gpus=$1
  local env=$2
  local ph1_omega=${3:-$PH1_OMEGA}
  local ph1_sigma=${4:-$PH1_SIGMA}
  local ph1_max_penalty_count=${5:-1}

  local tags="ph2,e3t,multi_penalty_max${ph1_max_penalty_count}"
  if [[ "$OLD_OVERCOOKED" == "1" ]]; then
    tags="${tags},old_overcooked"
  fi

  local -a cmd=("./run_user_wandb.sh"
    --gpus "$gpus"
    --seeds "$NUM_SEEDS"
    --seed "$FIXED_SEED"
    --env "$env"
    --exp "$EXP"
    --env-device "$ENV_DEVICE"
    --nenvs "$NENVS"
    --nsteps "$NSTEPS"
    --tags "$tags" \
    --ph1-beta $PH1_BETA \
    --ph1-beta-schedule-enabled $PH1_BETA_SCHEDULE_ENABLED \
    --ph1-beta-start $PH1_BETA_START \
    --ph1-beta-end $PH1_BETA_END \
    --ph1-beta-schedule-horizon-env-steps $PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS \
    --ph1-omega $ph1_omega \
    --ph1-sigma $ph1_sigma \
    --ph1-dist $PH1_DIST_THRESH \
    --ph1-pool-size $PH1_POOL_SIZE \
    --ph1-normal-prob $PH1_NORMAL_PROB \
    --ph1-multi-penalty-enabled $PH1_MULTI_PENALTY_ENABLED \
    --ph1-max-penalty-count $ph1_max_penalty_count \
    --ph1-multi-penalty-single-weight $PH1_MULTI_PENALTY_SINGLE_WEIGHT \
    --ph1-multi-penalty-other-weight $PH1_MULTI_PENALTY_OTHER_WEIGHT \
    --ph1-epsilon $PH1_EPSILON \
    --ph1-warmup-steps $PH1_WARMUP_STEPS \
    --ph1-eval-enabled $PH1_EVAL_ENABLED \
    --ph1-eval-every-env-steps $PH1_EVAL_EVERY_ENV_STEPS \
    --ph1-eval-video-every-env-steps $PH1_EVAL_VIDEO_EVERY_ENV_STEPS \
    --ph1-eval-defer-video $PH1_EVAL_DEFER_VIDEO \
    --ph1-eval-disable-jit $PH1_EVAL_DISABLE_JIT \
    --ph1-eval-offline-only $PH1_EVAL_OFFLINE_ONLY \
    --ph1-eval-log-video $PH1_EVAL_LOG_VIDEO \
    --ph1-eval-viz-episodes $PH1_EVAL_VIZ_EPISODES \
    --ph1-eval-num-seeds $PH1_EVAL_NUM_SEEDS \
    --ph2-ratio-stage1 $PH2_RATIO_STAGE1 \
    --ph2-ratio-stage2 $PH2_RATIO_STAGE2 \
    --ph2-ratio-stage3 $PH2_RATIO_STAGE3 \
    --shared-prediction "$SHARED_PREDICTION" \
    --action-prediction "$ACTION_PREDICTION" \
    --state-prediction "$STATE_PREDICTION" \
    --pred-z-dim "$PRED_Z_DIM" \
    --state-pred-loss-coef "$STATE_PRED_LOSS_COEF")

  if [[ "$OLD_OVERCOOKED" == "1" ]]; then
    cmd+=(--old-overcooked)
  fi

  if [[ -n "$PH2_FIXED_IND_PROB" ]]; then
    cmd+=(--ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB")
  fi
  if [[ -n "$PH2_EPSILON" ]]; then
    cmd+=(--ph2-epsilon "$PH2_EPSILON")
  fi

  echo "Executing: ${cmd[*]}"
  "${cmd[@]}"
}

# -----------------------------------------------------------------------------
# PH2 Preset: OV1 layouts / omega=10 / sigma=2.0 / max_penalty_count=1
# - Sequential execution by layout (one-by-one)
# -----------------------------------------------------------------------------
SWEEP_GPUS="0,1,2,3,4"
TARGET_OMEGA=10.0
TARGET_SIGMA=2.0
TARGET_MAX_COUNT=1

echo "[PH2-SWEEP] start: env=OV1_LAYOUTS gpus=$SWEEP_GPUS omega=$TARGET_OMEGA sigma=$TARGET_SIGMA max_penalty_count=$TARGET_MAX_COUNT"
if [[ "$OLD_OVERCOOKED" == "1" ]]; then
  echo "[PH2-SWEEP] engine=old_overcooked(v1)"
fi

# PH2 mode: state prediction + non-shared predictor
SHARED_PREDICTION="False"
ACTION_PREDICTION="True"
STATE_PREDICTION="False"
echo "[PH2-SWEEP] case=state_not_shared shared=$SHARED_PREDICTION action=$ACTION_PREDICTION state=$STATE_PREDICTION"


echo "[PH2-SWEEP] layout=counter_circuit"
run_ph2 "$SWEEP_GPUS" "counter_circuit" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

echo "[PH2-SWEEP] layout=cramped_room"
run_ph2 "$SWEEP_GPUS" "cramped_room" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

wait 

echo "[PH2-SWEEP] layout=asymm_advantages"
run_ph2 "$SWEEP_GPUS" "asymm_advantages" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

echo "[PH2-SWEEP] layout=coord_ring"
run_ph2 "$SWEEP_GPUS" "coord_ring" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

echo "[PH2-SWEEP] layout=forced_coord"
run_ph2 "$SWEEP_GPUS" "forced_coord" "$TARGET_OMEGA" "$TARGET_SIGMA" "$TARGET_MAX_COUNT"

echo "[PH2-SWEEP] all OV1 layout jobs finished."
