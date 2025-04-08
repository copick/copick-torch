#!/bin/bash

# IMPORTANT: This script needs to be run manually after merging the PR
# to remove redundant files.

echo "Removing redundant files after consolidating MONAI augmentations..."

# List of files to remove
FILES_TO_REMOVE=(
  "copick_torch/monai_augmentations.py"
  "tests/test_monai_augmentations.py"
  "examples/monai_augmentation_demo.py"
)

# Remove each file
for file in "${FILES_TO_REMOVE[@]}"; do
  if [ -f "$file" ]; then
    echo "Removing $file"
    rm -f "$file"
  else
    echo "File $file does not exist"
  fi
done

echo "Cleanup complete. Please commit these changes if needed."
