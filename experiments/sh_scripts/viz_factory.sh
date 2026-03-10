#!/usr/bin/env bash
# viz_factory.sh: 20251120부터 20251123까지의 runs 디렉토리에 대해 환경별로 그룹화하여 run_visualize.sh 실행

set -euo pipefail

# Change to script directory
cd "$(dirname "$0")" || exit 1

# run_visualize.sh 존재 확인
if [ ! -f "run_visualize.sh" ]; then
    echo "Error: run_visualize.sh not found in current directory"
    exit 1
fi

echo "Starting visualization factory by environment..."

# ./run_visualize.sh --gpu 7 --dir runs/20251209-000000_coord_ring --all --num_seeds 5
# ./run_visualize.sh --gpu 0 --dir runs/20260309-142621_genx798o_cramped_room_e3t --cross --num_seeds 5 &
# ./run_visualize.sh --gpu 1 --dir runs/20260309-171541_lomludhr_coord_ring_e3t --cross --num_seeds 5 &
# ./run_visualize.sh --gpu 2 --dir runs/20260309-203511_fjvm0sg0_forced_coord_e3t --cross --num_seeds 5 &
# ./run_visualize.sh --gpu 3 --dir runs/20260309-234546_6wfwbmfv_cramped_room_fcp --cross --num_seeds 5 &
# wait
# ./run_visualize.sh --gpu 0 --dir runs/20260309-235315_a229nt2i_counter_circuit_e3t --cross --num_seeds 5 &
# ./run_visualize.sh --gpu 1 --dir runs/20260310-011253_fao1ejiu_asymm_advantages_fcp --cross --num_seeds 5 & 
# ./run_visualize.sh --gpu 2 --dir runs/20260310-025845_beo670p0_coord_ring_fcp --cross --num_seeds 5 &
# # ./run_visualize.sh --gpu 2 --dir runs/20260310-042100_3yznlku8_asymm_advantages_e3t --cross --num_seeds 5 &
# ./run_visualize.sh --gpu 3 --dir runs/20260310-043022_nmirwdsr_forced_coord_fcp --cross --num_seeds 5 &
# wait
./run_visualize.sh --gpu 5 --dir runs/20260310-060205_y9a60a1c_counter_circuit_fcp --cross --num_seeds 5 &

echo ""
echo "All visualizations completed!"