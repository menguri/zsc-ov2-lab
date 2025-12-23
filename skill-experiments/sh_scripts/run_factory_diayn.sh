#!/bin/bash

cd "$(dirname "$0")" || exit 1

# ===============================
# DIAYN Experiment Factory Script
# ===============================
EXP="rnn-diayn"
ENV_DEVICE="cpu"
NENVS=64
NSTEPS=64

run_diayn() {
    local gpus=$1
    local env=$2
    local layout=$3
    local num_skills=$4
    local extra=$5
    local seeds=4
    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS \
        --num-skills $num_skills $extra \
        --seeds $seeds"
    if [ -n "$layout" ]; then
        cmd="$cmd --layout $layout"
    fi
    echo "Executing: $cmd"
    $cmd
}

# 실행 목록
# run_diayn "0,1,2,3,4" "grounded_coord_simple" "" 5 ""
# run_diayn "0,1,2,3,4" "grounded_coord_ring" "" 5 ""
# run_diayn "0,1,2,3,4" "demo_cook_simple" "" 5 ""
# run_diayn "0,1,2,3,4" "demo_cook_wide" "" 5 ""
# run_diayn "0,1,2,3,4" "test_time_simple" "" 5 ""
# run_diayn "0,1,2,3,4" "test_time_wide" "" 5 ""
# run_diayn "5,6" "cramped_room" "" 5 ""
# run_diayn "3,4" "asymm_advantages" "" 9 ""
# run_diayn "0,2,3,4,5" "asymm_advantages" "" 20 ""
# run_diayn "0,2,3,4,5" "asymm_advantages" "" 50 ""
# run_diayn "1,4" "asymm_advantages" "" 20 ""
# run_diayn "1,4" "asymm_advantages" "" 50 ""
# run_diayn "1,4" "grounded_coord_simple" "" 9 ""
# run_diayn "1,4" "grounded_coord_simple" "" 20 ""
run_diayn "1,2,4,5" "grounded_coord_simple" "" 100 ""
# run_diayn "5,6" "coord_ring" "" 5 ""
# run_diayn "5,6" "forced_coord" "" 5 ""
# run_diayn "1,2,4,5,6" "counter_circuit" "" 5 ""
