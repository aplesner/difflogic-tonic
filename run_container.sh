#!/bin/bash
set -e

export WANDB_CACHE_DIR=${SCRATCH_STORAGE_DIR:-./scratch}/wandb_cache

if [ -n "$SINGULARITY_CONTAINER_SCRATCH" ]; then
    # Run the script to sync the container to scratch if it doesn't exist
    bash helper_scripts/remote_sync_container.sh

    singularity run --nv --bind $PROJECT_STORAGE_DIR,$SCRATCH_STORAGE_DIR $SINGULARITY_CONTAINER_SCRATCH
else
    # exit if not sourced the environment setup script
    exit 1
fi
