#!/bin/bash

# Default values
RUN_DIR=""
SKILL=""
ALL_CKPT=""
NO_VIZ=""
SEED=42
NUM_SEEDS=1
GPU_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--dir)
      RUN_DIR="$2"
      shift 2
      ;;
    -s|--skill)
      SKILL="$2"
      shift 2
      ;;
    --all_ckpt)
      ALL_CKPT="--all_ckpt"
      shift
      ;;
    --no_viz)
      NO_VIZ="--no_viz"
      shift
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --num_seeds)
      NUM_SEEDS="$2"
      shift 2
      ;;
    -g|--gpu)
      GPU_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "$RUN_DIR" ]; then
  echo "Usage: $0 -d <run_directory> [-s <skill_idx>] [--all_ckpt] [--no_viz] [--seed <seed>] [--num_seeds <num>] [-g <gpu_id>]"
  exit 1
fi

# Construct command
# Assuming we are running from the root of the repo (where skill_ov2_experiments is a module)
CMD="python -m skill_ov2_experiments.ppo.utils.visualize_diayn --d $RUN_DIR --seed $SEED --num_seeds $NUM_SEEDS $ALL_CKPT $NO_VIZ"

if [ -n "$SKILL" ]; then
  CMD="$CMD --skill $SKILL"
fi

if [ -n "$GPU_ID" ]; then
  CMD="CUDA_VISIBLE_DEVICES=$GPU_ID $CMD"
fi

echo "Running: $CMD"
eval $CMD
