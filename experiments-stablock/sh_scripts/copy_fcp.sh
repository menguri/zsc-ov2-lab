#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <run_directory>"
  exit 1
fi

# Set the source directory
RUN="$1"

# Set the destination directory
DEST_DIR="fcp_populations/$RUN"

# Create the destination directory
mkdir -p $DEST_DIR

# Initialize variables
counter=0
subdir_counter=0

# Loop over the runs
for run in "runs/$RUN"/run_*; do
  # Check if we need to create a new subdirectory
  if (( counter % 8 == 0 )); then
    subdir_counter=$((subdir_counter + 1))
    mkdir -p "$DEST_DIR/fcp_$subdir_counter"
  fi
  
  # Copy the run to the appropriate subdirectory
  cp -r "$run" "$DEST_DIR/fcp_$subdir_counter/"
  
  # Increment the counter
  counter=$((counter + 1))
done
