#!/bin/bash

# Change to experiments directory
cd "$(dirname "$0")/.." || exit 1

if [ -z "$1" ]; then
  echo "Usage: $0 <run_directory>"
  exit 1
fi

run_dir="$1"

echo "Run directory: $run_dir"


python overcooked_v2_experiments/ppo/utils/visualize_ppo.py --num_seeds 2 --no_reset --d "$run_dir"
python overcooked_v2_experiments/ppo/utils/visualize_ppo.py --no_viz --cross --num_seeds 1000 --no_reset --d "$run_dir"
