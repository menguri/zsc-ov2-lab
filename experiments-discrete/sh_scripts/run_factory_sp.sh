#!/bin/bash

cd "$(dirname "$0")" || exit 1

# ===============================
# SP Experiment Factory Script
# ===============================
EXP="rnn-sp"
ENV_DEVICE="cpu"
NENVS=256
NSTEPS=256

run_sp() {
    local gpus=$1
    local env=$2
    local layout=$3
    local extra=$4
    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS $extra"
    if [ -n "$layout" ]; then
        cmd="$cmd --layout $layout"
    fi
    echo "Executing: $cmd"
    $cmd
}

# 실행 목록
# run_sp "0,1,2,3,4" "grounded_coord_simple" "" 
# run_sp "0,1,2,3,4" "grounded_coord_ring" "" 
# run_sp "0,1,2,3,4" "demo_cook_simple" "" 
# run_sp "0,1,2,3,4" "demo_cook_wide" "" 
# run_sp "0,1,2,3,4" "test_time_simple" "" 
# run_sp "0,1,2,3,4" "test_time_wide" "" 
# run_sp "5,6" "cramped_room" "" 
run_sp "1,2,4,5,6" "asymm_advantages" "" 
# run_sp "5,6" "coord_ring" "" 
# run_sp "5,6" "forced_coord" "" 
run_sp "1,2,4,5,6" "counter_circuit" "" 
