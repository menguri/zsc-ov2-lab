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

# E3D Latent Sensitivity Analysis
echo "[Task] Running Sensitivity Analysis on E3D run..."
# ./run_visualize.sh \
#     --gpu 7 \
#     --dir runs/20260119-043641_m3qkw9es_grounded_coord_simple_e3d_false \
#     --num_seeds 5 --no_viz \
#     --sensitivity

# ./run_visualize.sh --gpu 0 --dir runs/20260119-120428_x1r8v6an_grounded_coord_simple_e3d_false --num_seeds 5 --no_viz &
# ./run_visualize.sh --gpu 1 --dir runs/20260119-120429_2sowhq85_counter_circuit_e3d_false --num_seeds 5 --no_viz &
# ./run_visualize.sh --gpu 2 --dir runs/20260119-120429_ertl7cgf_counter_circuit_e3d_true --num_seeds 5 --no_viz & 
# ./run_visualize.sh --gpu 3 --dir runs/20260119-120429_yuntfo34_grounded_coord_simple_e3d_true --num_seeds 5 --no_viz & 
# ./run_visualize.sh --gpu 4 --dir runs/20260119-120428_x1r8v6an_grounded_coord_simple_e3d_false --num_seeds 5 --cross &
# ./run_visualize.sh --gpu 5 --dir runs/20260119-120429_2sowhq85_counter_circuit_e3d_false --num_seeds 5 --cross &
# ./run_visualize.sh --gpu 6 --dir runs/20260119-120429_ertl7cgf_counter_circuit_e3d_true --num_seeds 5 --cross & 
# ./run_visualize.sh --gpu 7 --dir runs/20260119-120429_yuntfo34_grounded_coord_simple_e3d_true --num_seeds 5 --cross & 
# wait
# ./run_visualize.sh --gpu 0 --dir runs/20260120-012856_idd3nzzh_grounded_coord_simple_fcp --num_seeds 5 --no_viz &
# ./run_visualize.sh --gpu 1 --dir runs/20260120-012857_joz8hjoy_grounded_coord_simple_fcp --num_seeds 5 --no_viz &
./run_visualize.sh --gpu 2 --dir runs/20260120-045414_3kvgh17o_counter_circuit_fcp --num_seeds 5 --no_viz & 
./run_visualize.sh --gpu 3 --dir runs/20260120-045414_vd6ec50y_counter_circuit_fcp --num_seeds 5 --no_viz & 
# ./run_visualize.sh --gpu 4 --dir runs/20260120-012856_idd3nzzh_grounded_coord_simple_fcp --num_seeds 5 --cross &
# ./run_visualize.sh --gpu 5 --dir runs/20260120-012857_joz8hjoy_grounded_coord_simple_fcp --num_seeds 5 --cross &
./run_visualize.sh --gpu 6 --dir runs/20260120-045414_3kvgh17o_counter_circuit_fcp --num_seeds 5 --cross & 
./run_visualize.sh --gpu 7 --dir runs/20260120-045414_vd6ec50y_counter_circuit_fcp --num_seeds 5 --cross &  

# ./run_visualize.sh --gpu 0 --dir runs/20260119-120428_x1r8v6an_grounded_coord_simple_e3d_false --num_seeds 5 --no_viz --sensitivity &
# ./run_visualize.sh --gpu 1 --dir runs/20260119-120429_2sowhq85_counter_circuit_e3d_false --num_seeds 5 --no_viz --sensitivity &
# ./run_visualize.sh --gpu 2 --dir runs/20260119-120429_ertl7cgf_counter_circuit_e3d_true --num_seeds 5 --no_viz --sensitivity & 
# ./run_visualize.sh --gpu 3 --dir runs/20260119-120429_yuntfo34_grounded_coord_simple_e3d_true --num_seeds 5 --no_viz --sensitivity & 
# ./run_visualize.sh --gpu 4 --dir runs/20260120-012856_idd3nzzh_grounded_coord_simple_fcp --num_seeds 5 --no_viz --sensitivity &
# ./run_visualize.sh --gpu 5 --dir runs/20260120-012857_joz8hjoy_grounded_coord_simple_fcp --num_seeds 5 --no_viz --sensitivity &
./run_visualize.sh --gpu 0 --dir runs/20260120-045414_3kvgh17o_counter_circuit_fcp --num_seeds 5 --no_viz --sensitivity & 
./run_visualize.sh --gpu 1 --dir runs/20260120-045414_vd6ec50y_counter_circuit_fcp --num_seeds 5 --no_viz --sensitivity & 
echo ""
echo "All visualizations completed!"