#!/usr/bin/env bash
# ====== OvercookedV2 확신 트리거(UC/OC) 일괄 실행 스크립트 ======
# - 상단: under-confidence(과도한 불확실 구간) 대응 프로필(sp_uc)
# - 하단: over-confidence(과신) 대응 프로필(sp_oc)
# 공통 옵션: --env-device cpu --nenvs 256 --nsteps 256 --seeds 10
# 필요 시 GPU 목록/seed 수 등을 직접 수정해서 사용하세요.

set -euo pipefail

# Change to script directory
cd "$(dirname "$0")" || exit 1

GPU_SET="3,4,5,6,7"
COMMON_OPTS=(--env-device cpu --nenvs 256 --nsteps 256 --seeds 10)

# Confidence trigger parameters (override defaults in config files)
UC_THRESHOLD="0.7"
UC_N_THRESHOLD="50"
OC_THRESHOLD="0.3"
OC_N_THRESHOLD="100"

# --------------------------------------------------------------------
# UC (sp_uc) : layout 순회 실행
# --------------------------------------------------------------------
./run_user_wandb.sh --gpus "${GPU_SET}" --env grounded_coord_simple --exp rnn-sp \
    "${COMMON_OPTS[@]}" --conf-profile sp_uc --conf-threshold "${UC_THRESHOLD}" --conf-n-threshold "${UC_N_THRESHOLD}"
./run_user_wandb.sh --gpus "${GPU_SET}" --env grounded_coord_ring --exp rnn-sp \
    "${COMMON_OPTS[@]}" --conf-profile sp_uc --conf-threshold "${UC_THRESHOLD}" --conf-n-threshold "${UC_N_THRESHOLD}"
./run_user_wandb.sh --gpus "${GPU_SET}" --env demo_cook_simple --exp rnn-sp \
    "${COMMON_OPTS[@]}" --conf-profile sp_uc --conf-threshold "${UC_THRESHOLD}" --conf-n-threshold "${UC_N_THRESHOLD}"
./run_user_wandb.sh --gpus "${GPU_SET}" --env demo_cook_wide --exp rnn-sp \
    "${COMMON_OPTS[@]}" --conf-profile sp_uc --conf-threshold "${UC_THRESHOLD}" --conf-n-threshold "${UC_N_THRESHOLD}"
./run_user_wandb.sh --gpus "${GPU_SET}" --env test_time_simple --exp rnn-sp \
    "${COMMON_OPTS[@]}" --conf-profile sp_uc --conf-threshold "${UC_THRESHOLD}" --conf-n-threshold "${UC_N_THRESHOLD}"
./run_user_wandb.sh --gpus "${GPU_SET}" --env test_time_wide --exp rnn-sp \
    "${COMMON_OPTS[@]}" --conf-profile sp_uc --conf-threshold "${UC_THRESHOLD}" --conf-n-threshold "${UC_N_THRESHOLD}"

# --------------------------------------------------------------------
# OC (sp_oc) : 동일 레이아웃 반복 실행
# --------------------------------------------------------------------
./run_user_wandb.sh --gpus "${GPU_SET}" --env grounded_coord_simple --exp rnn-sp \
    "${COMMON_OPTS[@]}" --conf-profile sp_oc --conf-threshold "${OC_THRESHOLD}" --conf-n-threshold "${OC_N_THRESHOLD}"
./run_user_wandb.sh --gpus "${GPU_SET}" --env grounded_coord_ring --exp rnn-sp \
    "${COMMON_OPTS[@]}" --conf-profile sp_oc --conf-threshold "${OC_THRESHOLD}" --conf-n-threshold "${OC_N_THRESHOLD}"
./run_user_wandb.sh --gpus "${GPU_SET}" --env demo_cook_simple --exp rnn-sp \
    "${COMMON_OPTS[@]}" --conf-profile sp_oc --conf-threshold "${OC_THRESHOLD}" --conf-n-threshold "${OC_N_THRESHOLD}"
./run_user_wandb.sh --gpus "${GPU_SET}" --env demo_cook_wide --exp rnn-sp \
    "${COMMON_OPTS[@]}" --conf-profile sp_oc --conf-threshold "${OC_THRESHOLD}" --conf-n-threshold "${OC_N_THRESHOLD}"
./run_user_wandb.sh --gpus "${GPU_SET}" --env test_time_simple --exp rnn-sp \
    "${COMMON_OPTS[@]}" --conf-profile sp_oc --conf-threshold "${OC_THRESHOLD}" --conf-n-threshold "${OC_N_THRESHOLD}"
./run_user_wandb.sh --gpus "${GPU_SET}" --env test_time_wide --exp rnn-sp \
    "${COMMON_OPTS[@]}" --conf-profile sp_oc --conf-threshold "${OC_THRESHOLD}" --conf-n-threshold "${OC_N_THRESHOLD}"
    
echo "[confidence-ucoc] 모든 레이아웃 실행 커맨드 준비 완료"
