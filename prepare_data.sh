#!/bin/bash

# Data preparation script
# Usage: ./prepare_data.sh <config_file>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 config_nmnist.yaml"
    exit 1
fi

CONFIG_FILE=$1

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

echo "=== Data Preparation Script ==="
echo "Config: $CONFIG_FILE"
echo "SCRATCH_STORAGE_DIR: ${SCRATCH_STORAGE_DIR:-not set (using default 'storage/')}"
echo ""

# Add current directory to Python path for src/ imports
export PYTHONPATH="${PYTHONPATH}:."

# Run data preparation
python3 prepare_data.py "$CONFIG_FILE"

# Check return status
if [ $? -ne 0 ]; then
    echo "Data preparation failed!"
    exit 1
fi

echo ""
echo "=== Final Storage Status ==="

# Show storage directory sizes if they exist
if [ -n "$SCRATCH_STORAGE_DIR" ] && [ -d "$SCRATCH_STORAGE_DIR/storage" ]; then
    echo "Scratch storage contents:"
    ls -lh "$SCRATCH_STORAGE_DIR/storage/"
elif [ -d "storage" ]; then
    echo "Local storage contents:"
    ls -lh storage/
fi

echo ""
echo "Data preparation complete! You can now run training scripts."