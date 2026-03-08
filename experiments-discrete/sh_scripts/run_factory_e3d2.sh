#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1

# ==============================================================================
# E3D Experiment Factory Script (with FCP partner)
# ==============================================================================

# Common Configuration
EXP="rnn-e3d"
ENV_DEVICE="cpu"
NENVS=128
NSTEPS=128

# E3D Specific Settings
EPSILON=0.1
EGO_DIR=True
USE_PM=True
HISTORY_LEN=3
GRU_HIDDEN_DIM=128

# Function to get FCP path based on env
get_fcp_path() {
    local env=$1
    case $env in
        "grounded_coord_simple")
            echo "fcp_populations/grounded_coord_simple_sp"
            ;;
        "counter_circuit")
            echo "fcp_populations/counter_circuit_sp"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to run experiment
run_e3d() {
    local gpus=$1
    local env=$2
    local layout=$3
    local ego_dir=${4:-$EGO_DIR}
    local history_len=${5:-$HISTORY_LEN}
    
    echo "================================================================================"
    echo "STARTING E3D EXPERIMENT"
    echo "ENV: $env, LAYOUT: $layout"
    echo "GPUS: $gpus, EGO_DIR: $ego_dir, HIST_LEN: $history_len"
    echo "================================================================================"
    
    local fcp_path
    fcp_path=$(get_fcp_path "$env")

    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS \
        --e3d-epsilon $EPSILON \
        --ego-direction $ego_dir \
        --history-len $history_len \
        --tags e3d \
        model.GRU_HIDDEN_DIM=$GRU_HIDDEN_DIM \
        USE_PARTNER_MODELING=$USE_PM \
        wandb.name=e3d_${ego_dir}_${EPSILON}_${history_len}"

    if [ -n "$fcp_path" ]; then
        cmd="$cmd --fcp $fcp_path"
    fi
        
    if [ -n "$layout" ]; then
        cmd="$cmd --layout $layout"
    fi
    
    echo "Executing: $cmd"
    $cmd
    
    echo "================================================================================"
    echo "FINISHED E3D EXPERIMENT"
    echo "================================================================================"
    echo ""
}

# ==============================================================================
# Execution List
# Usage: run_e3d <GPUS> <ENV_GROUP> <LAYOUT>
# ==============================================================================

# Grounded Coord Simple (True/False)
run_e3d "0,1,2,3,4" "grounded_coord_simple" "" True &
run_e3d "3,4,5,6,7" "grounded_coord_simple" "" False &
wait
# Counter Circuit (True/False)
run_e3d "0,1,2,3,4" "counter_circuit" "" True &
run_e3d "3,4,5,6,7" "counter_circuit" "" False &