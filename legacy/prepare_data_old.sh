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
echo "SCRATCH_STORAGE_DIR: ${SCRATCH_STORAGE_DIR:-not set (using default './scratch/')}"
echo ""

# Use singularity if SINGULARITY_CONTAINER is set and bind the project and scratch directories
if [ -n "$SINGULARITY_CONTAINER" ]; then
    echo "Using Singularity container: $SINGULARITY_CONTAINER"
    mkdir -p ${SCRATCH_STORAGE_DIR:-./scratch/}

    SINGULARITY_CMD="singularity exec --nv --bind $PROJECT_STORAGE_DIR,$SCRATCH_STORAGE_DIR $SINGULARITY_CONTAINER"
    $SINGULARITY_CMD python3 prepare_data.py "$CONFIG_FILE"
else
    # Add current directory to Python path for src/ imports
    export PYTHONPATH="${PYTHONPATH}:."

    # Run data preparation
    python3 prepare_data.py "$CONFIG_FILE"
fi

# Check return status
if [ $? -ne 0 ]; then
    echo "Data preparation failed!"
    exit 1
fi

echo ""
echo "Data preparation complete! You can now run training scripts."