#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || exit 1


# ====== OvercookedV2 - layout별 실행 명령 (GPU:0) ======

# --------------------------------------------------------------------
# original — layout: cramped_room  → CNN ONLY
# --------------------------------------------------------------------
# ./run_user_wandb.sh --gpus 6 --env original --layout cramped_room --exp cnn --env-device cpu --nenvs 128 --nsteps 128
# # sa (iterations=10)
# ./run_user_wandb.sh --gpus 6 --env original --layout cramped_room --exp cnn --iterations 10 --env-device cpu --nenvs 128 --nsteps 128
# ./run_user_wandb.sh --gpus 6 --env original --layout cramped_room --exp rnn-op --env-device cpu --nenvs 256 --nsteps 256
# --------------------------------------------------------------------
# grounded_coord_simple — 고정 레이아웃
# --------------------------------------------------------------------
# # sp
# ./run_user_wandb.sh --gpus 6,7 --env grounded_coord_simple --exp rnn-sp --env-device cpu --nenvs 256 --nsteps 256
# panic-sp (partner random action window)
# ./run_user_wandb.sh --gpus 6,7 --env grounded_coord_simple --exp panic-sp --env-device cpu --nenvs 256 --nsteps 256 --panic --panic-start 50 --panic-duration 30
# # op
# ./run_user_wandb.sh --gpus 4 --env grounded_coord_simple --exp rnn-op --env-device cpu --nenvs 256 --nsteps 256
# sa (iterations=10)
# ./run_user_wandb_for_st.sh --gpus 4 --env grounded_coord_simple --exp rnn-sa --iterations 10 --env-device cpu --nenvs 256 --nsteps 256
# fcp (population 준비 필요: --fcp 경로 교체) - FCP는 메모리 사용량이 높아 nenvs 64로 감소
# ./run_user_wandb.sh --gpus 3,4 --env grounded_coord_simple --exp rnn-fcp --fcp fcp_populations/grounded_coord_simple_avs-2-sp --env-device cpu --nenvs 256 --nsteps 256 --fcp-device "$FCP_DEVICE"

# --------------------------------------------------------------------
# grounded_coord_ring — 고정 레이아웃
# --------------------------------------------------------------------
# ./run_user_wandb.sh --gpus 2,3,4,6,7 --env grounded_coord_ring --exp rnn-sp --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 4 --env grounded_coord_ring --exp rnn-sa --iterations 10 --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 4 --env grounded_coord_ring --exp rnn-op --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 0 --env grounded_coord_ring --exp rnn-fcp --fcp fcp_populations/grounded_coord_ring --env-device cpu --nenvs 256 --nsteps 256 --fcp-device "$FCP_DEVICE"

# --------------------------------------------------------------------
# demo_cook_simple — 고정 레이아웃
# --------------------------------------------------------------------
# ./run_user_wandb.sh --gpus 6,7 --env demo_cook_simple --exp rnn-sp --env-device cpu --nenvs 256 --nsteps 256
# # panic-sp
# ./run_user_wandb.sh --gpus 6,7 --env demo_cook_simple --exp panic-sp --env-device cpu --nenvs 256 --nsteps 256 --panic --panic-start 50 --panic-duration 30
# ./run_user_wandb.sh --gpus 3 --env demo_cook_simple --exp rnn-sa --iterations 10 --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 3 --env demo_cook_simple --exp rnn-op --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 5 --env demo_cook_simple --exp rnn-fcp --fcp fcp_populations/demo_cook_simple_avs-2-sp --env-device cpu --nenvs 256 --nsteps 256 --fcp-device "$FCP_DEVICE"

# --------------------------------------------------------------------
# demo_cook_wide — 고정 레이아웃
# --------------------------------------------------------------------
# ./run_user_wandb.sh --gpus 0,1 --env demo_cook_wide --exp rnn-sp --env-device cpu --nenvs 256 --nsteps 256
./run_user_wandb.sh --gpus 0,1 --env demo_cook_wide --exp rnn-e3t --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 3 --env demo_cook_wide --exp rnn-sa --iterations 10 --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 3 --env demo_cook_wide --exp rnn-op --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 0 --env demo_cook_wide --exp rnn-fcp --fcp fcp_populations/demo_cook_wide --env-device cpu --nenvs 256 --nsteps 256 --fcp-device "$FCP_DEVICE"

# # --------------------------------------------------------------------
# # test_time_simple — 고정 레이아웃
# # --------------------------------------------------------------------
# ./run_user_wandb.sh --gpus 6,7 --env test_time_simple --exp rnn-sp --env-device cpu --nenvs 256 --nsteps 256
# # panic-sp
# ./run_user_wandb.sh --gpus 6,7 --env test_time_simple --exp panic-sp --env-device cpu --nenvs 256 --nsteps 256 --panic --panic-start 50 --panic-duration 30
# ./run_user_wandb.sh --gpus 2 --env test_time_simple --exp rnn-sa --iterations 10 --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 2 --env test_time_simple --exp rnn-op --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 3,4 --env test_time_simple --exp rnn-fcp --fcp fcp_populations/test_time_simple_avs-2-sp --env-device cpu --nenvs 256 --nsteps 256

# # --------------------------------------------------------------------
# # test_time_wide — 고정 레이아웃
# # --------------------------------------------------------------------
# ./run_user_wandb.sh --gpus 2,3,4,6,7 --env test_time_wide --exp rnn-sp --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 2 --env test_time_wide --exp rnn-sa --iterations 5 --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 2 --env test_time_wide --exp rnn-op --env-device cpu --nenvs 256 --nsteps 256
# ./run_user_wandb.sh --gpus 0 --env test_time_wide --exp rnn-fcp --fcp fcp_populations/test_time_wide --env-device cpu --nenvs 256 --nsteps 256