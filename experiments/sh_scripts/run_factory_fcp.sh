#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1

# ==============================================================================
# FCP Experiment Factory Script
# Runs FCP experiments sequentially on different layouts.
# ==============================================================================

# Common Configuration
EXP="rnn-fcp"
ENV_DEVICE="cpu"
NENVS=256
NSTEPS=256

# FCP Specific Settings
FCP_DEVICE="gpu"
SEEDS=10

# Function to get FCP path based on env
get_fcp_path() {
    local env=$1
    case $env in
        "grounded_coord_simple")
            echo "fcp_populations/grounded_coord_simple_avs-2-256-sp"
            ;;
        "grounded_coord_ring")
            echo "fcp_populations/grounded_coord_ring_sp"
            ;;
        "demo_cook_simple")
            echo "fcp_populations/demo_cook_simple_avs-2-256-sp"
            ;;
        "demo_cook_wide")
            echo "fcp_populations/demo_cook_wide_sp"
            ;;
        "test_time_simple")
            echo "fcp_populations/test_time_simple_avs-2-256-sp"
            ;;
        "test_time_wide")
            echo "fcp_populations/test_time_wide_sp"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to run experiment
run_fcp() {
    local gpus=$1
    local env=$2
    local layout=$3
    
    echo "================================================================================"
    echo "STARTING FCP EXPERIMENT"
    echo "ENV: $env, LAYOUT: $layout"
    echo "GPUS: $gpus"
    echo "================================================================================"
    
    local fcp_path=$(get_fcp_path $env)
    
    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS \
        --seeds $SEEDS \
        --fcp-device $FCP_DEVICE"
        
    if [ -n "$fcp_path" ]; then
        cmd="$cmd --fcp $fcp_path"
    fi
        
    if [ -n "$layout" ]; then
        cmd="$cmd --layout $layout"
    fi
    
    echo "Executing: $cmd"
    $cmd
    
    echo "================================================================================"
    echo "FINISHED FCP EXPERIMENT"
    echo "================================================================================"
    echo ""
}

# ==============================================================================
# Execution List (Uncomment lines to run)
# ==============================================================================

# 1. Grounded Coord Simple
run_fcp "0,1,2,3,4" "grounded_coord_simple" ""

# 2. Grounded Coord Ring
run_fcp "0,1,2,3,4" "grounded_coord_ring" ""

# 3. Demo Cook Simple
run_fcp "0,1,2,3,4" "demo_cook_simple" ""

# 4. Demo Cook Wide
run_fcp "0,1,2,3,4" "demo_cook_wide" ""

# 5. Test Time Simple
run_fcp "0,1,2,3,4" "test_time_simple" ""

# 6. Test Time Wide
run_fcp "0,1,2,3,4" "test_time_wide" ""

# 7. Cramped Room (Original) - No FCP path
run_fcp "0,1,2,3,4" "cramped_room" ""

# 8. Asymmetric Advantages (Original) - No FCP path
run_fcp "0,1,2,3,4" "asymm_advantages" ""

# 9. Coordination Ring (Original) - No FCP path
run_fcp "0,1,2,3,4" "coord_ring" ""

# 10. Forced Coordination (Original) - No FCP path
run_fcp "0,1,2,3,4" "forced_coord" ""

# 11. Counter Circuit (Original) - No FCP path
run_fcp "0,1,2,3,4" "counter_circuit" ""