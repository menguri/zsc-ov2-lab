#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1

# ==============================================================================
# OP Experiment Factory Script
# Runs OP experiments sequentially on different layouts.
# ==============================================================================

# Common Configuration
EXP="rnn-op"
ENV_DEVICE="cpu"
NENVS=256
NSTEPS=256

# Function to run experiment
run_op() {
    local gpus=$1
    local env=$2
    local layout=$3
    
    echo "================================================================================"
    echo "STARTING OP EXPERIMENT"
    echo "ENV: $env, LAYOUT: $layout"
    echo "GPUS: $gpus"
    echo "================================================================================"
    
    local cmd="./run_user_wandb.sh \
        --gpus $gpus \
        --env $env \
        --exp $EXP \
        --env-device $ENV_DEVICE \
        --nenvs $NENVS \
        --nsteps $NSTEPS"
        
    if [ -n "$layout" ]; then
        cmd="$cmd --layout $layout"
    fi
    
    echo "Executing: $cmd"
    $cmd
    
    echo "================================================================================"
    echo "FINISHED OP EXPERIMENT"
    echo "================================================================================"
    echo ""
}

# ==============================================================================
# Execution List (Uncomment lines to run)
# ==============================================================================

# # 1. Grounded Coord Simple
# run_op "0,1,2,3,4" "grounded_coord_simple" ""

# # 2. Grounded Coord Ring
# run_op "0,1,2,3,4" "grounded_coord_ring" ""

# 3. Demo Cook Simple
run_op "0,1,2,3,4" "demo_cook_simple" ""

# # 4. Demo Cook Wide
# run_op "0,1,2,3,4" "demo_cook_wide" ""

# # 5. Test Time Simple
# run_op "0,1,2,3,4" "test_time_simple" ""

# # 6. Test Time Wide
# run_op "0,1,2,3,4" "test_time_wide" ""

# # 7. Cramped Room (Original)
# run_op "6,7" "cramped_room" ""

# # 8. Asymmetric Advantages (Original)
# run_op "6,7" "asymm_advantages" ""

# # 9. Coordination Ring (Original)
# run_op "6,7" "coord_ring" ""

# # 10. Forced Coordination (Original)
# run_op "6,7" "forced_coord" ""

# # 11. Counter Circuit (Original)
# run_op "6,7" "counter_circuit" ""





# # --------------------------------------------------------------------
# ./run_user_wandb.sh --gpus 7 --env test_time_simple --exp rnn-sp --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 2 --env test_time_simple --exp rnn-sa --iterations 10 --env-device cpu --nenvs 128 --nsteps 128
# ./run_user_wandb.sh --gpus 4,5 --env test_time_simple --exp rnn-op --env-device cpu --nenvs 128 --nsteps 128
# # # ./run_user_wandb.sh --gpus 0 --env test_time_simple --exp rnn-fcp --fcp fcp_populations/test_time_simple --env-device cpu --nenvs 64 --nsteps 128

# # --------------------------------------------------------------------
# # test_time_wide — 고정 레이아웃
# # --------------------------------------------------------------------
# ./run_user_wandb.sh --gpus 7 --env test_time_wide --exp rnn-sp --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 2 --env test_time_wide --exp rnn-sa --iterations 5 --env-device cpu --nenvs 128 --nsteps 128
# ./run_user_wandb.sh --gpus 4,5 --env test_time_wide --exp rnn-op --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 0 --env test_time_wide --exp rnn-fcp --fcp fcp_populations/test_time_wide --env-device cpu --nenvs 64 --nsteps 128