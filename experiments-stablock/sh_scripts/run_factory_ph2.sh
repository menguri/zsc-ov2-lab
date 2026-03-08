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
: "${PH1_EPSILON:=0.2}"
: "${PH2_EPSILON:=0.2}"
: "${PH1_WARMUP_STEPS:=2000000}"
: "${PH1_CONTRASTIVE_ENABLED:=False}"
: "${PH1_CONTRASTIVE_COEF:=0.05}"
: "${PH1_CONTRASTIVE_TEMP:=0.1}"
: "${PH1_CONTRASTIVE_PROJ_DIM:=64}"
: "${PH1_CONTRASTIVE_ENTRY_DROPOUT:=0.05}"
: "${PH1_CONTRASTIVE_MULTI_POS:=True}"
: "${PH1_CONTRASTIVE_DENOM_TRAIN_MASK:=True}"
: "${PH1_EVAL_ENABLED:=False}"
: "${PH1_EVAL_EVERY_ENV_STEPS:=1000000}"
: "${PH1_EVAL_VIDEO_EVERY_ENV_STEPS:=1000000}"
: "${PH1_EVAL_DEFER_VIDEO:=True}"
: "${PH1_EVAL_DISABLE_JIT:=True}"
: "${PH1_EVAL_OFFLINE_ONLY:=True}"
: "${PH1_EVAL_LOG_VIDEO:=False}"
: "${PH1_EVAL_VIZ_EPISODES:=0}"
: "${PH1_EVAL_NUM_SEEDS:=1}"

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

  local tags="ph2,e3t"
  if [[ "${PH1_CONTRASTIVE_ENABLED}" == "True" || "${PH1_CONTRASTIVE_ENABLED}" == "true" || "${PH1_CONTRASTIVE_ENABLED}" == "1" ]]; then
    tags="${tags},contrastive"
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
    --ph1-epsilon $PH1_EPSILON \
    --ph1-warmup-steps $PH1_WARMUP_STEPS \
    --ph1-contrastive-enabled $PH1_CONTRASTIVE_ENABLED \
    --ph1-contrastive-coef $PH1_CONTRASTIVE_COEF \
    --ph1-contrastive-temp $PH1_CONTRASTIVE_TEMP \
    --ph1-contrastive-proj-dim $PH1_CONTRASTIVE_PROJ_DIM \
    --ph1-contrastive-entry-dropout $PH1_CONTRASTIVE_ENTRY_DROPOUT \
    --ph1-contrastive-multi-pos $PH1_CONTRASTIVE_MULTI_POS \
    --ph1-contrastive-denom-train-mask $PH1_CONTRASTIVE_DENOM_TRAIN_MASK \
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
    --ph2-ratio-stage3 $PH2_RATIO_STAGE3)

  if [[ -n "$PH2_FIXED_IND_PROB" ]]; then
    cmd+=(--ph2-fixed-ind-prob "$PH2_FIXED_IND_PROB")
  fi
  if [[ -n "$PH2_EPSILON" ]]; then
    cmd+=(--ph2-epsilon "$PH2_EPSILON")
  fi

  echo "Executing: ${cmd[*]}"
  "${cmd[@]}"
}

# Example:
# PH1_CONTRASTIVE_ENABLED=True PH2_EPSILON=0.2 \
#   ./run_factory_ph2.sh

# -----------------------------------------------------------------------------
# PH2 Sweep
# - Envs: OV1 only (exclude counter_circuit)
# - Omega: 10.0 fixed
# - Sigma: 2.0 fixed
# - Sequential execution on the same GPU set to avoid overlap/OOM
# -----------------------------------------------------------------------------
SWEEP_GPUS="0,1,2,3,4"
SWEEP_ENVS=("asymm_advantages" "coord_ring" "cramped_room" "forced_coord")
SWEEP_OMEGAS=(10.0)
SWEEP_SIGMAS=(2.0)

echo "[PH2-SWEEP] start: gpus=$SWEEP_GPUS"

for env in "${SWEEP_ENVS[@]}"; do
  for omega in "${SWEEP_OMEGAS[@]}"; do
    for sigma in "${SWEEP_SIGMAS[@]}"; do
      echo "[PH2-SWEEP] env=$env omega=$omega sigma=$sigma"
      # Sweep blocked-target parameters for PH1 latent penalty terms.
      run_ph2 "$SWEEP_GPUS" "$env" "$omega" "$sigma"
    done
  done
done

echo "[PH2-SWEEP] all jobs finished."
