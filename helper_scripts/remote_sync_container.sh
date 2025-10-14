#!/bin/bash

if [ ! -f "$SINGULARITY_CONTAINER_SCRATCH" ]; then
    echo "Copying singularity container to scratch space from $SINGULARITY_CONTAINER_PROJECT..."
    mkdir -p "$SINGULARITY_CONTAINER_SCRATCH"   
    rsync -ah --temp-dir="$SINGULARITY_SCRATCH_STORAGE_DIR" "$SINGULARITY_CONTAINER_PROJECT" "$SINGULARITY_CONTAINER_SCRATCH"
fi