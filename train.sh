#!/bin/bash
set -e

export WANDB_CACHE_DIR=${SCRATCH_STORAGE_DIR:-./scratch}/wandb_cache

if [ -n "$SINGULARITY_CONTAINER_SCRATCH" ]; then
    # Run the script to sync the container to scratch if it doesn't exist
    bash helper_scripts/remote_sync_container.sh

    singularity exec --nv --bind $PROJECT_STORAGE_DIR,$SCRATCH_STORAGE_DIR $SINGULARITY_CONTAINER_SCRATCH python3 main.py "$@"
else
    export PYTHONPATH="${PYTHONPATH}:."
    python3 main.py "$@"
fi
