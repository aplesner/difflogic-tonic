#!/bin/bash
set -e

if [ -n "$SINGULARITY_CONTAINER_SCRATCH" ]; then
    bash helper_scripts/remote_sync_container.sh

    singularity exec --nv --bind $PROJECT_STORAGE_DIR,$SCRATCH_STORAGE_DIR $SINGULARITY_CONTAINER_SCRATCH python3 prepare_data.py "$@"
else
    export PYTHONPATH="${PYTHONPATH}:."
    python3 prepare_data.py "$@"
fi
