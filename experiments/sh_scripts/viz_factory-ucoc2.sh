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

# echo "Processing grounded_coord_simple..."
# ./run_visualize.sh --gpu 7 --dir runs/20251202-154142_61jl3kgp_grounded_coord_simple_sp-uc --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 7 --dir runs/20251202-171540_bt0bj552_grounded_coord_ring_sp-uc --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 7 --dir runs/20251202-185323_5u2vdi34_demo_cook_simple_sp-uc --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 7 --dir runs/20251202-203203_7gl7xle9_demo_cook_wide_sp-uc --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 7 --dir runs/20251202-221019_j0q1azlp_test_time_simple_sp-uc --all --num_seeds 10 --no_viz

# ./run_visualize.sh --gpu 7 --dir runs/20251202-154856_uht1r5gd_grounded_coord_simple_sp-oc --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 6,7 --dir runs/20251202-173021_v7icf8i8_grounded_coord_ring_sp-oc --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 6,7 --dir runs/20251202-191801_qky8z7qd_demo_cook_simple_sp-oc --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 6,7 --dir runs/20251202-210035_vfax9jyv_demo_cook_wide_sp-oc --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 6,7 --dir runs/20251202-225125_nksu58ho_test_time_simple_sp-oc --all --num_seeds 10 --no_viz

./run_visualize.sh --gpu 3,4 --dir runs/20251202-234751_kel03nlz_test_time_wide_sp-uc --all --num_seeds 10 --no_viz
./run_visualize.sh --gpu 3,4 --dir runs/20251203-003016_ernioxit_test_time_wide_sp-oc --all --num_seeds 10 --no_viz




echo ""
echo "All visualizations completed!"