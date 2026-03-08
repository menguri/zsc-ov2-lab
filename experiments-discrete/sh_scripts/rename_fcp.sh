#!/bin/bash

# Base directory where the target directories are located
if [ -z "$1" ]; then
  echo "Usage: $0 <run_directory>"
  exit 1
fi

# Set the source directory
base_dir="$1"

# Counter for naming the new directories
i=0

# Iterate over each directory in the base directory
for dir in "$base_dir"/*; do
    # Check if it is a directory and contains 'run_0'
    if [ -d "$dir" ] && [ -d "$dir/run_0" ]; then
        # Set the destination directory name
        dest="$base_dir/run_$i"

        # Check if the destination directory already exists to avoid overwriting
        if [ ! -d "$dest" ]; then
            # Copy the run_0 directory to the new destination
            cp -r "$dir/run_0" "$dest"
            echo "Copied $dir/run_0 to $dest"
            # Increment the counter for the next directory
            i=$((i + 1))
        else
            echo "Directory $dest already exists, skipping."
        fi

        # Stop the loop if we have created 10 directories
        if [ $i -ge 10 ]; then
            break
        fi
    fi
done
