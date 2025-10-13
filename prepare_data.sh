#!/bin/bash
set -e

if [ -n "$SINGULARITY_CONTAINER_SCRATCH" ]; then
    [ ! -f "$SINGULARITY_CONTAINER_SCRATCH" ] && rsync -ah "$SINGULARITY_CONTAINER_PROJECT" "$SINGULARITY_CONTAINER_SCRATCH"
    mkdir -p ${SCRATCH_STORAGE_DIR:-./scratch}
    singularity exec --nv --bind $PROJECT_STORAGE_DIR,$SCRATCH_STORAGE_DIR $SINGULARITY_CONTAINER_SCRATCH python3 prepare_data.py "$@"
else
    export PYTHONPATH="${PYTHONPATH}:."
    python3 prepare_data.py "$@"
fi
