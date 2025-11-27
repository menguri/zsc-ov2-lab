#!/usr/bin/env bash
# ====== OvercookedV2 panic-sp 전용 실행 스크립트 ======
# 각 레이아웃별 panic 윈도우 실험만 남김.
# 공통 옵션: --nenvs 256 --nsteps 256 --panic --panic-start 50 --panic-duration 30 --env-device cpu
# 필요 시 GPU 할당 조정 가능.

set -euo pipefail

# Change to script directory
cd "$(dirname "$0")" || exit 1

# grounded_coord_simple
./run_user_wandb.sh --gpus 0,1,2,3,4 --env grounded_coord_simple --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 30 --panic-duration 10

./run_user_wandb.sh --gpus 0,1,2,3,4 --env grounded_coord_simple --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 30 --panic-duration 5

./run_user_wandb.sh --gpus 0,1,2,3,4 --env grounded_coord_simple --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 60 --panic-duration 10

# # grounded_coord_ring
./run_user_wandb.sh --gpus 0,1,2,3,4 --env grounded_coord_ring --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 30 --panic-duration 10

./run_user_wandb.sh --gpus 0,1,2,3,4 --env grounded_coord_ring --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 30 --panic-duration 5

./run_user_wandb.sh --gpus 0,1,2,3,4 --env grounded_coord_ring --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 60 --panic-duration 10

# # demo_cook_simple
./run_user_wandb.sh --gpus 0,1,2,3,4 --env demo_cook_simple --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 30 --panic-duration 10

./run_user_wandb.sh --gpus 0,1,2,3,4 --env demo_cook_simple --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 30 --panic-duration 5

./run_user_wandb.sh --gpus 0,1,2,3,4 --env demo_cook_simple --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 60 --panic-duration 10

# # demo_cook_wide
./run_user_wandb.sh --gpus 0,1,2,3,4 --env demo_cook_wide --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 30 --panic-duration 10

./run_user_wandb.sh --gpus 0,1,2,3,4 --env demo_cook_wide --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 30 --panic-duration 5

./run_user_wandb.sh --gpus 0,1,2,3,4 --env demo_cook_wide --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 60 --panic-duration 10

# test_time_simple
./run_user_wandb.sh --gpus 0,1,2,3,4 --env test_time_simple --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 30 --panic-duration 10

./run_user_wandb.sh --gpus 0,1,2,3,4 --env test_time_simple --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 30 --panic-duration 5

./run_user_wandb.sh --gpus 0,1,2,3,4 --env test_time_simple --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 60 --panic-duration 10

# # test_time_wide
./run_user_wandb.sh --gpus 0,1,2,3,4 --env test_time_wide --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 30 --panic-duration 10

./run_user_wandb.sh --gpus 0,1,2,3,4 --env test_time_wide --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 30 --panic-duration 5

./run_user_wandb.sh --gpus 0,1,2,3,4 --env test_time_wide --exp panic-sp \
	--env-device cpu --nenvs 256 --nsteps 256 \
	--panic --panic-start 60 --panic-duration 10

echo "[panic-sp] 모든 레이아웃 실행 커맨드 준비 완료"