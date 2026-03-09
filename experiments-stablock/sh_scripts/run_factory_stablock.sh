#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1

# ==============================================================================
# Stablock Experiment Factory Script
# ======================================================================

# Common Configuration
EXP="rnn-stablock"
ENV_DEVICE="cpu"
NENVS=128
NSTEPS=128

# Stablock Specific Settings
STABLOCK_ENABLED=True
STABLOCK_PENALTY=30.0
STABLOCK_NO_BLOCK_PROB=0.5

# E3T Settings
EPSILON=0.1
USE_PM=True
PRED_COEF=1.0

run_stablock() {
    local gpus=$1
    local env=$2
    local layout=$3
    local penalty=$4

    # If penalty not provided, fall back to a default
    if [ -z "$penalty" ]; then
        penalty=$STABLOCK_PENALTY
    fi

    echo "================================================================================"
    echo "STARTING STABLOCK EXPERIMENT"
    echo "ENV: $env, LAYOUT: $layout"
    echo "GPUS: $gpus"
    echo "STABLOCK_PENALTY: $penalty"
    echo "================================================================================"

    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS \
        --e3t-epsilon $EPSILON \
        --tags stablock,e3t \
        --stablock-enabled $STABLOCK_ENABLED \
        --stablock-penalty $penalty \
        --stablock-no-block-prob $STABLOCK_NO_BLOCK_PROB \
        --use-partner-modeling $USE_PM \
        --pred-loss-coef $PRED_COEF"

    if [ -n "$layout" ]; then
        cmd="$cmd --layout $layout"
    fi

    echo "Executing: $cmd"
    $cmd

    echo "================================================================================"
    echo "FINISHED STABLOCK EXPERIMENT"
    echo "================================================================================"
    echo ""
}

# ==============================================================================
# Execution List
# Usage: run_stablock <GPUS> <ENV_GROUP> <LAYOUT>
# ==============================================================================

# run_stablock "0,1" "grounded_coord_simple" "" "30.0" &
# sleep 5
# run_stablock "2,3" "grounded_coord_simple" "" "5.0" &
# sleep 5
# run_stablock "4,5" "grounded_coord_simple" "" "0.5" &
# run_stablock "0,1" "counter_circuit" "" "5.0" &
# sleep 5
# run_stablock "2,3" "counter_circuit" "" "0.5" &
# sleep 5
run_stablock "2,3" "grounded_coord_simple" "" "30" &
# wait
# run_stablock "0,1" "forced_coord" "" "30" &