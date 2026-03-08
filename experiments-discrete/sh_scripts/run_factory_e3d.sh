#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1

# ==============================================================================
# E3D Experiment Factory Script
# Runs E3D experiments sequentially on different layouts.
# ==============================================================================

# Common Configuration
EXP="rnn-e3d"
ENV_DEVICE="cpu"
NENVS=128
NSTEPS=128

# E3D Specific Settings
EPSILON=0.3
EGO_DIR=True
USE_PM=True
HISTORY_LEN=5    # Default 5
GRU_HIDDEN_DIM=64

# Function to run experiment
run_e3d() {
    local gpus=$1
    local env=$2
    local layout=$3
    local ego_dir=${4:-$EGO_DIR}  # Use provided ego_dir or default to EGO_DIR
    local history_len=${5:-$HISTORY_LEN} # Use provided history_len or default
    
    echo "================================================================================"
    echo "STARTING E3D EXPERIMENT"
    echo "ENV: $env, LAYOUT: $layout"
    echo "GPUS: $gpus, EGO_DIR: $ego_dir, HIST_LEN: $history_len"
    echo "================================================================================"
    
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
# Execution List (Uncomment lines to run)
# Usage: run_e3d <GPUS> <ENV_GROUP> <LAYOUT>
# ==============================================================================

# 1. Grounded Coord Simple
run_e3d "0,1" "grounded_coord_simple" "" True &
run_e3d "2,3" "grounded_coord_simple" "" False &
# # 2. Grounded Coord Ring
# run_e3d "0,1,3,4,5" "grounded_coord_ring" ""

# # 3. Demo Cook Simple
# run_e3d "0,1,3,4,5" "demo_cook_simple" ""

# # 4. Demo Cook Wide
# run_e3d "0,1,2,3,4" "demo_cook_wide" ""

# # 5. Test Time Simple
# run_e3d "0,1,3,4,5" "test_time_simple" ""

# # 6. Test Time Wide
# run_e3d "5,6" "test_time_wide" ""

# # 7. Cramped Room (Original)
# run_e3d "0,1,2,3,4" "cramped_room" ""

# # 8. Asymmetric Advantages (Original)
# run_e3d "0,1,3,4,5" "asymm_advantages" ""

# # 9. Coordination Ring (Original)
# run_e3d "5,6" "coord_ring" ""

# # 10. Forced Coordination (Original)
# run_e3d "1,3,4,6,7" "forced_coord" ""

# # 11. Counter Circuit (Original)
# run_e3d "3,4,5,6,7" "counter_circuit" ""
run_e3d "4,5" "counter_circuit" "" True&
run_e3d "6,7" "counter_circuit" "" False &