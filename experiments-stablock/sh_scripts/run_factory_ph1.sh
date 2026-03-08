#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -z "$REPO_ROOT" || ! -d "$REPO_ROOT/overcooked_v2_experiments" ]]; then
  REPO_ROOT="/home/mlic/mingukang/ex-overcookedv2/experiments-stablock"
fi

# ==============================================================================
# STA-PH1 Experiment Factory Script
# ==============================================================================

# Common Configuration
EXP="rnn-ph1"
ENV_DEVICE="cpu"
NENVS=64
NSTEPS=128
NUM_SEEDS=5

# PH1 Default Settings
BETA=0.5 # Default value, can be overridden
PH1_BETA_SCHEDULE_ENABLED=True
PH1_BETA_START=0.0
PH1_BETA_END=2.0
PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS=-1
OMEGA=1.0 # Default value, can be overridden
SIGMA=1.0 # Default value, can be overridden
DIST_THRESH=0.1
POOL_SIZE=128
NORMAL_PROB=0.5
PH1_EPSILON=0.2 # With probability epsilon, one agent takes random action
WARMUP_STEPS=1000000 # 5m steps for warm-up (normal interaction)
PH1_EVAL_ENABLED=False # 학습 중 eval/snapshot 비활성화, 종료 후 버퍼 기반 일괄 export 사용
PH1_EVAL_EVERY_ENV_STEPS=200000 # 오프라인 eval cadence 기준
PH1_EVAL_VIDEO_EVERY_ENV_STEPS=1000000 # video 주기: 1m env steps
PH1_EVAL_DEFER_VIDEO=True # 학습 중엔 snapshot만 저장, 비디오는 오프라인 생성
PH1_EVAL_VIZ_EPISODES=0
PH1_EVAL_NUM_SEEDS=1
PH1_EVAL_DISABLE_JIT=True
PH1_EVAL_OFFLINE_ONLY=True
PH1_EVAL_LOG_VIDEO=True  # 학습 후 오프라인 비디오 생성 on/off 스위치
PH1_POST_TRAIN_OFFLINE=False # 학습/평가 완전 분리: 학습 후 별도 eval 스크립트로 실행
PH1_OFFLINE_VIZ_MAX_STEPS=400

# Contrastive Settings (override 가능: VAR=... ./sh_scripts/run_factory_ph1.sh)
: "${PH1_CONTRASTIVE_ENABLED:=True}"
: "${PH1_CONTRASTIVE_COEF:=0.05}"
: "${PH1_CONTRASTIVE_TEMP:=0.1}"
: "${PH1_CONTRASTIVE_PROJ_DIM:=64}"
: "${PH1_CONTRASTIVE_ENTRY_DROPOUT:=0.05}"
: "${PH1_CONTRASTIVE_MULTI_POS:=True}"
: "${PH1_CONTRASTIVE_DENOM_TRAIN_MASK:=True}"

# E3T Settings
EPSILON=0.0 # 사용자 요청으로 0.0

run_ph1() {
    local gpus=$1
    local env=$2
    local beta=$3
    local omega=$4
    local normal_prob=$5
    local ph1_epsilon=$6
    local warmup_steps=$7
    local sigma=$8
    local beta_end=$9

    # Defaults check
    if [ -z "$beta" ]; then beta=$BETA; fi
    if [ -z "$omega" ]; then omega=$OMEGA; fi
    if [ -z "$normal_prob" ]; then normal_prob=$NORMAL_PROB; fi
    if [ -z "$ph1_epsilon" ]; then ph1_epsilon=$PH1_EPSILON; fi
    if [ -z "$warmup_steps" ]; then warmup_steps=$WARMUP_STEPS; fi
    if [ -z "$sigma" ]; then sigma=$SIGMA; fi
    if [ -z "$beta_end" ]; then beta_end=$PH1_BETA_END; fi

    echo "================================================================================"
    echo "STARTING STA-PH1 EXPERIMENT"
    echo "ENV: $env"
    echo "GPUS: $gpus"
    echo "BETA: $beta, OMEGA: $omega, NORMAL_PROB: $normal_prob, EPSILON: $ph1_epsilon"
    echo "BETA_SCHEDULE: enabled=$PH1_BETA_SCHEDULE_ENABLED start=$PH1_BETA_START end=$beta_end horizon=$PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS"
    echo "SIGMA: $sigma"
    echo "WARMUP: $warmup_steps"
    echo "CONTRASTIVE: enabled=$PH1_CONTRASTIVE_ENABLED coef=$PH1_CONTRASTIVE_COEF temp=$PH1_CONTRASTIVE_TEMP proj_dim=$PH1_CONTRASTIVE_PROJ_DIM entry_dropout=$PH1_CONTRASTIVE_ENTRY_DROPOUT multi_pos=$PH1_CONTRASTIVE_MULTI_POS denom_train_mask=$PH1_CONTRASTIVE_DENOM_TRAIN_MASK"
    echo "================================================================================"

    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --seeds $NUM_SEEDS \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS \
        --e3t-epsilon $EPSILON \
        --tags ph1,e3t \
        --ph1-beta $beta \
        --ph1-beta-schedule-enabled $PH1_BETA_SCHEDULE_ENABLED \
        --ph1-beta-start $PH1_BETA_START \
        --ph1-beta-end $beta_end \
        --ph1-beta-schedule-horizon-env-steps $PH1_BETA_SCHEDULE_HORIZON_ENV_STEPS \
        --ph1-omega $omega \
        --ph1-sigma $sigma \
        --ph1-dist $DIST_THRESH \
        --ph1-pool-size $POOL_SIZE \
        --ph1-normal-prob $normal_prob \
        --ph1-epsilon $ph1_epsilon \
        --ph1-warmup-steps $warmup_steps \
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
        --ph1-eval-num-seeds $PH1_EVAL_NUM_SEEDS"

    echo "Executing: $cmd"
    local run_log_file
    run_log_file=$(mktemp)
    $cmd | tee "$run_log_file"

    if [[ "$PH1_POST_TRAIN_OFFLINE" == "True" || "$PH1_POST_TRAIN_OFFLINE" == "1" ]]; then
      local run_base_dir
      run_base_dir="$(grep -m 1 '^run_dir ' "$run_log_file" | awk '{print $2}' | tr -d '\r' || true)"
      if [[ -z "$run_base_dir" ]]; then
        run_base_dir="$(grep -m 1 -E '^\[RUNDBG\] run_base_dir = ' "$run_log_file" | awk -F ' = ' '{print $2}' | tr -d '\r' || true)"
      fi
      if [[ -n "$run_base_dir" && ! "$run_base_dir" = /* ]]; then
        run_base_dir="$REPO_ROOT/$run_base_dir"
      fi
      if [[ -n "$run_base_dir" && "$run_base_dir" = /runs/* && ! -d "$run_base_dir" ]]; then
        run_base_dir="$REPO_ROOT$run_base_dir"
      fi
      rm -f "$run_log_file"

      if [[ -z "$run_base_dir" ]]; then
        echo "[WARN] offline eval skipped: run_base_dir not found in log."
      elif [[ ! -d "$run_base_dir/eval" ]]; then
        echo "[WARN] offline eval skipped: snapshot directory missing: $run_base_dir/eval"
      else
        echo "[INFO] Start post-training PH1 offline eval: $run_base_dir"
        local offline_cmd=(python3 "$REPO_ROOT/overcooked_v2_experiments/ppo/ph1_offline_video.py" --run-base-dir "$run_base_dir")
        offline_cmd+=(--eval-every-env-steps "$PH1_EVAL_EVERY_ENV_STEPS")
        offline_cmd+=(--video-every-env-steps "$PH1_EVAL_VIDEO_EVERY_ENV_STEPS")
        offline_cmd+=(--viz-max-steps "$PH1_OFFLINE_VIZ_MAX_STEPS")
        offline_cmd+=(--log-video "$PH1_EVAL_LOG_VIDEO")
        (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT:${PYTHONPATH}" "${offline_cmd[@]}")
      fi
    else
      rm -f "$run_log_file"
      echo "[INFO] Post-train offline eval is disabled. Run separately:"
      echo "       ./sh_scripts/run_offline_eval.sh ph1 <run_base_dir>"
    fi

    echo "================================================================================"
    echo "FINISHED STA-PH1 EXPERIMENT"
    echo "================================================================================"
    echo ""
}

# ==============================================================================
# Execution List
# Usage:
#   run_ph1 <GPUS> <ENV> <BETA> <OMEGA> <NORMAL_PROB> <PH1_EPSILON> <WARMUP_STEPS> <SIGMA> <BETA_END>
# Contrastive example:
#   PH1_CONTRASTIVE_ENABLED=True PH1_CONTRASTIVE_COEF=0.05 PH1_CONTRASTIVE_TEMP=0.1 PH1_CONTRASTIVE_PROJ_DIM=64 PH1_CONTRASTIVE_ENTRY_DROPOUT=0.05 PH1_CONTRASTIVE_MULTI_POS=True PH1_CONTRASTIVE_DENOM_TRAIN_MASK=True ./sh_scripts/run_factory_ph1.sh
# Minimal toggle example:
#   PH1_CONTRASTIVE_ENABLED=True ./sh_scripts/run_factory_ph1.sh
#
# Policy:
# - no for-loop
# - sequential code blocks
# - two GPU groups in parallel: 0,1,2 and 3,4,5
# - each run uses NUM_SEEDS=3
# ==============================================================================

# grounded_coord_simple: (omega,sigma) = (10,0.5), beta_end sweep = 0.5/1.0/1.5/2.0
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "10.0" "0.0" "" "" "3.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "10.0" "0.0" "" "" "2.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "10.0" "0.0" "" "" "1.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "10.0" "0.0" "" "" "2.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "20.0" "0.0" "" "" "1.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "20.0" "0.0" "" "" "2.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "counter_circuit" "1.0" "10.0" "0.0" "" "" "1.0" "1.0" &
# wait
run_ph1 "0,1,2,3,4" "counter_circuit" "1.0" "10.0" "0.0" "" "" "3.0" "1.0" &
wait
# run_ph1 "0,1,2,3,4" "counter_circuit" "1.0" "20.0" "0.0" "" "" "1.0" "1.0" &
# wait
run_ph1 "0,1,2,3,4" "counter_circuit" "1.0" "20.0" "0.0" "" "" "3.0" "1.0" &
wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "20.0" "0.0" "" "" "3.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "grounded_coord_simple" "1.0" "20.0" "0.0" "" "" "2.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "counter_circuit" "1.0" "20.0" "0.0" "" "" "3.0" "1.0" &
# wait
# run_ph1 "0,1,2,3,4" "counter_circuit" "1.0" "20.0" "0.0" "" "" "2.0" "1.0" &
# wait
# run_ph1 "0,1,2" "grounded_coord_simple" "1.0" "10.0" "0.5" "" "" "0.5" "1.5" &
# run_ph1 "3,4,5" "grounded_coord_simple" "1.0" "10.0" "0.5" "" "" "0.5" "2.0" &
# wait

# # grounded_coord_simple: (omega,sigma) = (20,0.5), beta_end sweep = 0.5/1.0/1.5/2.0
# run_ph1 "0,1,2" "grounded_coord_simple" "1.0" "20.0" "0.5" "" "" "0.5" "0.5" &
# run_ph1 "3,4,5" "grounded_coord_simple" "1.0" "20.0" "0.5" "" "" "0.5" "1.0" &
# wait
# run_ph1 "0,1,2" "grounded_coord_simple" "1.0" "20.0" "0.5" "" "" "0.5" "1.5" &
# run_ph1 "3,4,5" "grounded_coord_simple" "1.0" "20.0" "0.5" "" "" "0.5" "2.0" &
# wait

# counter_circuit: (omega,sigma) = (90,0.5), beta_end sweep = 0.5/1.0/1.5/2.0
# run_ph1 "0,1,2" "counter_circuit" "1.0" "90.0" "0.5" "" "" "0.5" "0.5" &
# run_ph1 "3,4,5" "counter_circuit" "1.0" "90.0" "0.5" "" "" "0.5" "1.0" &
# wait
# run_ph1 "0,1,2" "counter_circuit" "1.0" "90.0" "0.5" "" "" "0.5" "1.5" &
# run_ph1 "3,4,5" "counter_circuit" "1.0" "90.0" "0.5" "" "" "0.5" "2.0" &
# wait

echo "All jobs finished."
