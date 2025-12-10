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
./run_visualize.sh --gpu 7 --dir runs/20251209-100533_9muxsfpp_forced_coord_stl --all --num_seeds 10 --no_viz
./run_visualize.sh --gpu 7 --dir runs/20251209-133945_wrtxyhl0_counter_circuit_stl --all --num_seeds 10 --no_viz

echo ""
echo "All visualizations completed!"