#!/bin/bash

# Change to experiments directory
cd "$(dirname "$0")/.." || exit 1

# Function to display usage information
usage() {
  echo "Usage: $0 [-p PATTERN] [-o]"
  echo "  -p PATTERN   Pattern to match the directories (e.g., 'runs/*')"
  echo "  -f           Overwrite the file exists check"
  exit 1
}

# Default values
OVERWRITE=false

# Parse command-line arguments
while getopts ":p:f" opt; do
  case ${opt} in
    p)
      PATTERN="$OPTARG*"
      ;;
    f)
      OVERWRITE=true
      ;;
    \?)
      echo "Invalid option: $OPTARG" 1>&2
      usage
      ;;
    :)
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      usage
      ;;
  esac
done

# Check if pattern is provided
if [ -z "$PATTERN" ]; then
  echo "Pattern is required."
  usage
fi

echo "Pattern: $PATTERN"

# Collect all directories to process
dirs_to_process=()

# Loop through each directory matching the pattern
for dir in $PATTERN; do
  echo "Checking directory: $dir"
  # Check if the directory exists and is a directory
  if [ -d "$dir" ]; then
    # Check if the directory is not empty
    if [ "$(ls -A $dir)" ]; then
      # Check if the file reward_summary_cross_plot.png does not exist or overwrite is true
      if [ ! -f "$dir/reward_summary_cross_plot.png" ] || [ "$OVERWRITE" = true ]; then
        dirs_to_process+=("$dir")
      fi
    fi
  fi
done

# Print summary of directories to be processed
echo "Directories to be processed:"
for dir in "${dirs_to_process[@]}"; do
  echo "$dir"
done

# Process each directory
for dir in "${dirs_to_process[@]}"; do
  # Execute the Python command
  python overcooked_v2_experiments/helper/visualise.py --no_viz --cross --num_seeds 1000 --no_reset --d "$dir"
done
