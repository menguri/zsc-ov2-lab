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

# # grounded_coord_simple
# echo "Processing grounded_coord_simple..."
# ./run_visualize.sh --gpu 1 --dir runs/20251120-065052_txtupjex_grounded_coord_simple_avs-2-256-op --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251120-065229_6e4kdnru_grounded_coord_simple_avs-2-256-st --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251121-012842_k0hq1tgt_grounded_coord_simple_avs-2-256-sp --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251126-073602_bw9rrc0t_grounded_coord_simple_avs-2-256-fcp --all --num_seeds 10 --no_viz


# # grounded_coord_ring
# echo "Processing grounded_coord_ring..."
./run_visualize.sh --gpu 6 --dir runs/20251128-024034_9t2oq9um_grounded_coord_ring_sp --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251123-134106_zma7dqet_grounded_coord_ring_avs-2-256-op --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251123-134505_t31pjfr9_grounded_coord_ring_avs-2-256-st --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251126-091833_n8hgdq5c_grounded_coord_ring_avs-2-256-fcp --all --num_seeds 10 --no_viz

# # demo_cook_simple
# echo "Processing demo_cook_simple..."
# ./run_visualize.sh --gpu 7 --dir runs/20251120-080837_pde5eo6h_demo_cook_simple_avs-2-256-sp --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251120-090142_wc7pazic_demo_cook_simple_avs-2-256-op --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 5 --dir runs/20251126-101817_2ty6yw06_demo_cook_simple_avs-2-256-fcp

# # demo_cook_wide
# echo "Processing demo_cook_wide..."
# ./run_visualize.sh --gpu 1 --dir runs/20251121-040307_ko8dwt00_demo_cook_wide_avs-2-256-op --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 6 --dir runs/20251121-040922_ii3s89wl_demo_cook_wide_avs-2-256-sp --num_seeds 10
# ./run_visualize.sh --gpu 1 --dir runs/20251126-111743_100fux5j_demo_cook_wide_avs-2-256-fcp --all --num_seeds 10 --no_viz

# # test_time_simple
# echo "Processing test_time_simple..."
# ./run_visualize.sh --gpu 1 --dir runs/20251120-085343_4ikg2dkw_test_time_simple_avs-2-256-st --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251120-092704_lxcvh421_test_time_simple_avs-2-256-sp --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251120-104220_ipwcsah4_test_time_simple_avs-2-256-sp --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251120-110537_o3fjide7_test_time_simple_avs-2-256-op --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251126-121729_fgbyx6wn_test_time_simple_avs-2-256-fcp --all --num_seeds 10 --no_viz

# # test_time_wide
# echo "Processing test_time_wide..."
./run_visualize.sh --gpu 6 --dir runs/20251128-025655_s497eete_test_time_wide_sp --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251121-060742_j5mcuaiw_test_time_wide_avs-2-256-op --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251126-131745_87hrfx3w_test_time_wide_avs-2-256-fcp --all --num_seeds 10 --no_viz


# # Panic experiment
# echo "Processing Panic experiment..."
# ./run_visualize.sh --gpu 1 --dir runs/20251123-065126_bp6nmar5_grounded_coord_simple_avs-2-panic-1 --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251123-080812_kmvv4ich_grounded_coord_simple_avs-2-panic-2 --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251123-092511_r5f02j17_grounded_coord_simple_avs-2-panic-3 --all --num_seeds 10 --no_viz

# ./run_visualize.sh --gpu 1 --dir runs/20251123-104158_div9xtdr_demo_cook_simple_avs-2-panic-1 --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251123-115810_bbwrfpbm_demo_cook_simple_avs-2-panic-2 --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251123-131346_pnec2dvl_demo_cook_simple_avs-2-panic-3 --all --num_seeds 10 --no_viz

# ./run_visualize.sh --gpu 1 --dir runs/20251123-142811_82rzusl3_test_time_simple_avs-2-panic-1 --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 1 --dir runs/20251123-154227_a95qa5b9_test_time_simple_avs-2-panic-2 --all --num_seeds 10 --no_viz
# ./run_visualize.sh --gpu 7 --dir runs/20251123-165718_ft3ykl8u_test_time_simple_avs-2-panic-3 --all --num_seeds 10 --no_viz



# ./run_visualize.sh --gpu 7 --dir runs/FCP_grounded_coord_simple_avs-2-256-sp_cc83466m_20251121-061204-1 --num_seeds 10


echo ""
echo "All visualizations completed!"