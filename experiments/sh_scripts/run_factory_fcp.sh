# # ====== OvercookedV2 - layout별 실행 명령 (GPU:0) ======

# Change to script directory
cd "$(dirname "$0")" || exit 1

# FCP_DEVICE 인자 처리 (기본값: cpu)
FCP_DEVICE=${1:-gpu}

# --------------------------------------------------------------------
# grounded_coord_simple — 고정 레이아웃
# --------------------------------------------------------------------
# fcp (population 준비 필요: --fcp 경로 교체) - FCP는 메모리 사용량이 높아 nenvs 64로 감소
# ./run_user_wandb.sh --gpus 0,1,2,3,4 --env grounded_coord_simple --exp rnn-fcp --fcp fcp_populations/grounded_coord_simple_avs-2-256-sp --env-device cpu --nenvs 256 --nsteps 256 --seeds 10 --fcp-device "$FCP_DEVICE"

# --------------------------------------------------------------------
# grounded_coord_ring — 고정 레이아웃
# --------------------------------------------------------------------
./run_user_wandb.sh --gpus 0,1,2,3,4 --env grounded_coord_ring --exp rnn-fcp --fcp fcp_populations/grounded_coord_ring_avs-2-256-sp --env-device cpu --nenvs 256 --nsteps 256 --seeds 10 --fcp-device "$FCP_DEVICE"

# --------------------------------------------------------------------
# demo_cook_simple — 고정 레이아웃
# --------------------------------------------------------------------
./run_user_wandb.sh --gpus 0,1,2,3,4 --env demo_cook_simple --exp rnn-fcp --fcp fcp_populations/demo_cook_simple_avs-2-256-sp --env-device cpu --nenvs 256 --nsteps 256 --seeds 10 --fcp-device "$FCP_DEVICE"

# --------------------------------------------------------------------
# demo_cook_wide — 고정 레이아웃
# --------------------------------------------------------------------
./run_user_wandb.sh --gpus 0,1,2,3,4 --env demo_cook_wide --exp rnn-fcp --fcp fcp_populations/demo_cook_wide_avs-2-256-sp --env-device cpu --nenvs 256 --nsteps 256 --seeds 10 --fcp-device "$FCP_DEVICE"

# --------------------------------------------------------------------
# test_time_simple — 고정 레이아웃
# --------------------------------------------------------------------
./run_user_wandb.sh --gpus 0,1,2,3,4 --env test_time_simple --exp rnn-fcp --fcp fcp_populations/test_time_simple_avs-2-256-sp --env-device cpu --nenvs 256 --nsteps 256 --seeds 10 --fcp-device "$FCP_DEVICE"

# --------------------------------------------------------------------
# test_time_wide — 고정 레이아웃
# --------------------------------------------------------------------
./run_user_wandb.sh --gpus 0,1,2,3,4 --env test_time_wide --exp rnn-fcp --fcp fcp_populations/test_time_wide_avs-2-256-sp --env-device cpu --nenvs 256 --nsteps 256 --seeds 10 --fcp-device "$FCP_DEVICE"