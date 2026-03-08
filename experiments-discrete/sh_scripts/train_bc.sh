#!/bin/bash

# Change to experiments directory
cd "$(dirname "$0")/.." || exit 1

# Define a list of config names
layouts=(
    "cramped_room"
    "asymm_advantages"
    "coord_ring"
    "forced_coord"
    "counter_circuit"
)

splits=(
    # "train"
    # "test"
    "all"
)

# Loop through the layouts and run the training script for each
for layout in "${layouts[@]}"; do
    for split in "${splits[@]}"; do
        echo "Training BC model for $layout layout with $split split"
        python overcooked_v2_experiments/human_rl/imitation/train_bc.py layouts=$layout SPLIT=$split
        echo "Finished training for $layout layout with $split split"
        echo "----------------------------------------"
    done
done
