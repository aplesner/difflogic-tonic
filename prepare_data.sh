#!/bin/bash

# Data preparation script with Hydra configuration
# Usage: ./prepare_data_hydra.sh [prepare=NAME] [KEY=VALUE...]
#
# Examples:
#   ./prepare_data_hydra.sh prepare=nmnist
#   ./prepare_data_hydra.sh prepare=cifar10dvs prepare/variants=events_20k
#   ./prepare_data_hydra.sh prepare=cifar10dvs data.reset_cache=true

set -e

echo "Starting Hydra data preparation..."
echo "Arguments: $@"
echo "SCRATCH_STORAGE_DIR: ${SCRATCH_STORAGE_DIR:-not set (using default './scratch/')}"

# Use singularity if SINGULARITY_CONTAINER is set
if [ -n "$SINGULARITY_CONTAINER" ]; then
    echo "Using Singularity container: $SINGULARITY_CONTAINER"

    if [ ! -f "$SINGULARITY_CONTAINER" ]; then
        echo "Error: Container not found: $SINGULARITY_CONTAINER"
        exit 1
    fi

    mkdir -p ${SCRATCH_STORAGE_DIR:-./scratch/}

    SINGULARITY_CMD="singularity exec --nv --bind $PROJECT_STORAGE_DIR,$SCRATCH_STORAGE_DIR $SINGULARITY_CONTAINER"
    $SINGULARITY_CMD python3 prepare_data_hydra.py "$@"
else
    # Add current directory to Python path for src/ imports
    export PYTHONPATH="${PYTHONPATH}:."

    python3 prepare_data_hydra.py "$@"
fi
