#!/usr/bin/env bash
# ====== OvercookedV2 sa 전용 실행 스크립트 (simple + 어려운 버전) ======
# 각 레이아웃별 sa 실험만 남김 (simple + 어려운 버전 모두 포함).
# 공통 옵션: --iterations 10 --env-device cpu --nenvs 256 --nsteps 256
# 필요 시 GPU 할당 조정 가능.

set -euo pipefail

# Change to script directory
cd "$(dirname "$0")" || exit 1

# grounded_coord_simple (simple)
./run_user_wandb.sh --gpus 1,2,5,6,7 --env grounded_coord_simple --exp rnn-sa --iterations 10 --env-device cpu --nenvs 256 --nsteps 256

# grounded_coord_ring (어려운 버전)
# ./run_user_wandb.sh --gpus 6,7 --env grounded_coord_ring --exp rnn-sa --iterations 10 --env-device cpu --nenvs 128 --nsteps 256 

# demo_cook_simple (simple)
# ./run_user_wandb.sh --gpus 6,7 --env demo_cook_simple --exp rnn-sa --iterations 10 --env-device cpu --nenvs 128 --nsteps 256

# demo_cook_wide (어려운 버전)
# ./run_user_wandb.sh --gpus 6,7 --env demo_cook_wide --exp rnn-sa --iterations 10 --env-device cpu --nenvs 128 --nsteps 256

# test_time_simple (simple)
# ./run_user_wandb_2.sh --gpus 1 --env test_time_simple --exp rnn-sa --iterations 10 --env-device cpu --nenvs 256 --nsteps 256

# test_time_wide (어려운 버전)
# ./run_user_wandb.sh --gpus 6,7 --env test_time_wide --exp rnn-sa --iterations 10 --env-device cpu --nenvs 128 --nsteps 256

echo "[sa] 모든 레이아웃 (simple + 어려운 버전) 실행 커맨드 준비 완료"