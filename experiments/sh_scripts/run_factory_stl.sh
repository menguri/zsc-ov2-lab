#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1

# ==============================================================================
# E3T Experiment Factory Script
# Runs E3T experiments sequentially on different layouts.
# ==============================================================================

# Common Configuration
EXP="rnn-e3t"
ENV_DEVICE="cpu"
NENVS=128
NSTEPS=128

# E3T Specific Settings
EPSILON=0.5
USE_PM=True
PRED_COEF=1.0

# Function to run experiment
run_e3t() {
    local gpus=$1
    local env=$2
    local layout=$3
    local anchor=$4  # 1=Enable STL, 0=Disable
    
    local anchor_arg=""
    local tag_arg=""
    
    if [ "$anchor" == "1" ]; then
        anchor_arg="--anchor"
        tag_arg="--tags e3t,stl"
    else
        tag_arg="--tags e3t"
    fi
    
    echo "================================================================================"
    echo "STARTING E3T EXPERIMENT"
    echo "ENV: $env, LAYOUT: $layout, ANCHOR: $anchor"
    echo "GPUS: $gpus"
    echo "================================================================================"
    
    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS \
        --e3t-epsilon $EPSILON \
        $anchor_arg \
        $tag_arg \
        USE_PARTNER_MODELING=$USE_PM \
        PRED_LOSS_COEF=$PRED_COEF"
        
    if [ -n "$layout" ]; then
        cmd="$cmd --layout $layout"
    fi
    
    echo "Executing: $cmd"
    $cmd
    
    echo "================================================================================"
    echo "FINISHED E3T EXPERIMENT"
    echo "================================================================================"
    echo ""
}

# ==============================================================================
# Execution List (Uncomment lines to run)
# Usage: run_e3t <GPUS> <ENV_GROUP> <LAYOUT> <ANCHOR(0/1)>
# ==============================================================================

# 1. Grounded Coord Simple
run_e3t "0,1,2,3,4" "grounded_coord_simple" "" "1"

# # 2. Grounded Coord Ring
run_e3t "0,1,2,3,4" "grounded_coord_ring" "" "1"

# # 3. Demo Cook Simple
run_e3t "0,1,2,3,4" "demo_cook_simple" "" "1"

# # 4. Demo Cook Wide
run_e3t "0,1,2,3,4" "demo_cook_wide" "" "1"

# # 5. Test Time Simple
run_e3t "0,1,2,3,4" "test_time_simple" "" "1"

# # 6. Test Time Wide
run_e3t "0,1,2,3,4" "test_time_wide" "" "1"

# 5. Cramped Room (Original)
run_e3t "0,1,2,3,4" "cramped_room" "" "1"

# 6. Asymmetric Advantages (Original)
run_e3t "0,1,2,3,4" "asymm_advantages" "" "1"

# # 7. Coordination Ring (Original)
# run_e3t "0,1,2,3,4" "coord_ring" "" "1"

# # 8. Forced Coordination (Original)
# run_e3t "0,1,2,3,4" "forced_coord" "" "1"

# # 9. Counter Circuit (Original)
# run_e3t "0,1,2,3,4" "counter_circuit" "" "1"
