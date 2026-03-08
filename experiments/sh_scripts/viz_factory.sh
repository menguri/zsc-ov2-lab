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
./run_visualize.sh --gpu 0 --dir runs/20260128-015425_9usv82dt_counter_circuit_fcp --cross --num_seeds 5

echo ""
echo "All visualizations completed!"