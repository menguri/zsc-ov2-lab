#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1

# ==============================================================================
# PSP Experiment Factory Script
# Runs PSP (Panic-SP) experiments sequentially on different layouts.
# ==============================================================================

# Common Configuration
EXP="panic-sp"
ENV_DEVICE="cpu"
NENVS=256
NSTEPS=256

# Function to run experiment
run_psp() {
    local gpus=$1
    local env=$2
    local layout=$3
    local panic_start=$4
    local panic_duration=$5
    
    echo "================================================================================"
    echo "STARTING PSP EXPERIMENT"
    echo "ENV: $env, LAYOUT: $layout"
    echo "PANIC START: $panic_start, DURATION: $panic_duration"
    echo "GPUS: $gpus"
    echo "================================================================================"
    
    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS \
        --panic \
        --panic-start $panic_start \
        --panic-duration $panic_duration"
        
    if [ -n "$layout" ]; then
        cmd="$cmd --layout $layout"
    fi
    
    echo "Executing: $cmd"
    $cmd
    
    echo "================================================================================"
    echo "FINISHED PSP EXPERIMENT"
    echo "================================================================================"
    echo ""
}

# ==============================================================================
# Execution List (Uncomment lines to run)
# ==============================================================================

# Grounded Coord Simple
run_psp "0,1,2,3,4" "grounded_coord_simple" "" 30 10
run_psp "0,1,2,3,4" "grounded_coord_simple" "" 30 5
run_psp "0,1,2,3,4" "grounded_coord_simple" "" 60 10

# Grounded Coord Ring
run_psp "0,1,2,3,4" "grounded_coord_ring" "" 30 10
run_psp "0,1,2,3,4" "grounded_coord_ring" "" 30 5
run_psp "0,1,2,3,4" "grounded_coord_ring" "" 60 10

# Demo Cook Simple
run_psp "0,1,2,3,4" "demo_cook_simple" "" 30 10
run_psp "0,1,2,3,4" "demo_cook_simple" "" 30 5
run_psp "0,1,2,3,4" "demo_cook_simple" "" 60 10

# Demo Cook Wide
run_psp "0,1,2,3,4" "demo_cook_wide" "" 30 10
run_psp "0,1,2,3,4" "demo_cook_wide" "" 30 5
run_psp "0,1,2,3,4" "demo_cook_wide" "" 60 10

# Test Time Simple
run_psp "0,1,2,3,4" "test_time_simple" "" 30 10
run_psp "0,1,2,3,4" "test_time_simple" "" 30 5
run_psp "0,1,2,3,4" "test_time_simple" "" 60 10

# Test Time Wide
run_psp "0,1,2,3,4" "test_time_wide" "" 30 10
run_psp "0,1,2,3,4" "test_time_wide" "" 30 5
run_psp "0,1,2,3,4" "test_time_wide" "" 60 10

# Cramped Room (Original)
run_psp "0,1,2,3,4" "cramped_room" "" 30 10
run_psp "0,1,2,3,4" "cramped_room" "" 30 5
run_psp "0,1,2,3,4" "cramped_room" "" 60 10

# Asymmetric Advantages (Original)
run_psp "0,1,2,3,4" "asymm_advantages" "" 30 10
run_psp "0,1,2,3,4" "asymm_advantages" "" 30 5
run_psp "0,1,2,3,4" "asymm_advantages" "" 60 10

# Coordination Ring (Original)
run_psp "0,1,2,3,4" "coord_ring" "" 30 10
run_psp "0,1,2,3,4" "coord_ring" "" 30 5
run_psp "0,1,2,3,4" "coord_ring" "" 60 10

# Forced Coordination (Original)
run_psp "0,1,2,3,4" "forced_coord" "" 30 10
run_psp "0,1,2,3,4" "forced_coord" "" 30 5
run_psp "0,1,2,3,4" "forced_coord" "" 60 10

# Counter Circuit (Original)
run_psp "0,1,2,3,4" "counter_circuit" "" 30 10
run_psp "0,1,2,3,4" "counter_circuit" "" 30 5
run_psp "0,1,2,3,4" "counter_circuit" "" 60 10