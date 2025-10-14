#!/bin/bash

echo "Syncing singularity container to scratch space from $SINGULARITY_PROJECT_STORAGE_DIR..."
mkdir -p "$SINGULARITY_SCRATCH_STORAGE_DIR"
rsync --info=progress3 -ah --temp-dir="$SINGULARITY_SCRATCH_STORAGE_DIR" "$SINGULARITY_CONTAINER_PROJECT" "$SINGULARITY_SCRATCH_STORAGE_DIR"