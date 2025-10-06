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

# Use singularity if SINGULARITY_CONTAINER is set and bind the project and scratch directories
if [ -n "$SINGULARITY_CONTAINER" ]; then
    echo "Using Singularity container: $SINGULARITY_CONTAINER"
    mkdir -p ${SCRATCH_STORAGE_DIR:-./scratch/}

    SINGULARITY_CMD="singularity exec --bind $PROJECT_STORAGE_DIR,$SCRATCH_STORAGE_DIR $SINGULARITY_CONTAINER"
    $SINGULARITY_CMD python3 main.py $CONFIG_FILE --job_id $JOB_ID $EXTRA_ARGS
else
    # Add current directory to Python path for src/ imports
    export PYTHONPATH="${PYTHONPATH}:."

    python3 main.py $CONFIG_FILE --job_id $JOB_ID $EXTRA_ARGS
fi