#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1

# ==============================================================================
# SA Experiment Factory Script
# Runs SA experiments sequentially on different layouts.
# ==============================================================================

# Common Configuration
EXP="rnn-sa"
ENV_DEVICE="cpu"
NENVS=256
NSTEPS=256

# SA Specific Settings
ITERATIONS=10

# Function to run experiment
run_sa() {
    local gpus=$1
    local env=$2
    local layout=$3
    
    echo "================================================================================"
    echo "STARTING SA EXPERIMENT"
    echo "ENV: $env, LAYOUT: $layout"
    echo "GPUS: $gpus"
    echo "================================================================================"
    
    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS \
        --iterations $ITERATIONS"
        
    if [ -n "$layout" ]; then
        cmd="$cmd --layout $layout"
    fi
    
    echo "Executing: $cmd"
    $cmd
    
    echo "================================================================================"
    echo "FINISHED SA EXPERIMENT"
    echo "================================================================================"
    echo ""
}

# ==============================================================================
# Execution List (Uncomment lines to run)
# ==============================================================================

# 1. Grounded Coord Simple
run_sa "0,1,2,3,4" "grounded_coord_simple" ""

# 2. Grounded Coord Ring
run_sa "0,1,2,3,4" "grounded_coord_ring" ""

# 3. Demo Cook Simple
run_sa "0,1,2,3,4" "demo_cook_simple" ""

# 4. Demo Cook Wide
run_sa "0,1,2,3,4" "demo_cook_wide" ""

# 5. Test Time Simple
run_sa "0,1,2,3,4" "test_time_simple" ""

# 6. Test Time Wide
run_sa "0,1,2,3,4" "test_time_wide" ""

# 7. Cramped Room (Original)
run_sa "0,1,2,3,4" "cramped_room" ""

# 8. Asymmetric Advantages (Original)
run_sa "0,1,2,3,4" "asymm_advantages" ""

# 9. Coordination Ring (Original)
run_sa "0,1,2,3,4" "coord_ring" ""

# 10. Forced Coordination (Original)
run_sa "0,1,2,3,4" "forced_coord" ""

# 11. Counter Circuit (Original)
run_sa "0,1,2,3,4" "counter_circuit" ""