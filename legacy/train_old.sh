#!/bin/bash

# Training script
# Usage: ./train.sh <config_file> <job_id>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_file> <job_id>"
    echo "Example: $0 config_nmnist.yaml job_001"
    exit 1
fi

CONFIG_FILE=$1
JOB_ID=$2
EXTRA_ARGS=${@:3}

echo "Starting training with config: $CONFIG_FILE, job_id: $JOB_ID"
echo "SCRATCH_STORAGE_DIR: ${SCRATCH_STORAGE_DIR:-not set (using default './scratch/')}"

export WANDB_CACHE_DIR=${SCRATCH_STORAGE_DIR:-./scratch/aplesner}/wandb_cache

# Use singularity if SINGULARITY_CONTAINER is set and bind the project and scratch directories
if [ -n "$SINGULARITY_CONTAINER" ]; then
    echo "Using Singularity container: $SINGULARITY_CONTAINER"

    # Sync container if needed (from project storage to local/scratch)
    if [ ! -f "$SINGULARITY_CONTAINER" ]; then
        echo "Container not found locally. Checking project storage..."

        # Try to find container in project storage
        CONTAINER_NAME=$(basename "$SINGULARITY_CONTAINER")
        PROJECT_CONTAINER="${PROJECT_STORAGE_DIR}/singularity/${CONTAINER_NAME}"

        if [ -f "$PROJECT_CONTAINER" ]; then
            echo "Syncing container from project storage..."
            CONTAINER_DIR=$(dirname "$SINGULARITY_CONTAINER")
            mkdir -p "$CONTAINER_DIR"
            rsync -avh --progress "$PROJECT_CONTAINER" "$SINGULARITY_CONTAINER"
            echo "Container synced successfully"
        else
            echo "Error: Container not found in project storage: $PROJECT_CONTAINER"
            exit 1
        fi
    else
        echo "Container found: $SINGULARITY_CONTAINER"
    fi

    mkdir -p ${SCRATCH_STORAGE_DIR:-./scratch/}

    SINGULARITY_CMD="singularity exec --nv --bind $PROJECT_STORAGE_DIR,$SCRATCH_STORAGE_DIR $SINGULARITY_CONTAINER"
    $SINGULARITY_CMD python3 main.py $CONFIG_FILE --job_id $JOB_ID $EXTRA_ARGS
else
    # Add current directory to Python path for src/ imports
    export PYTHONPATH="${PYTHONPATH}:."

    python3 main.py $CONFIG_FILE --job_id $JOB_ID $EXTRA_ARGS
fi