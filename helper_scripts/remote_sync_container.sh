#!/bin/bash

# Check that the PROJECT_NAME is sourced
if [ "$PROJECT_NAME" != "difflogic-tonic" ]; then
    echo "helper_scripts/project_variables.sh is not sourced"
    exit 1
fi

# If the timestamps do not differ do nothing
# if [ -f "$SINGULARITY_CONTAINER_SCRATCH" ] && [ "$SINGULARITY_CONTAINER_PROJECT" -ot "$SINGULARITY_CONTAINER_SCRATCH" ]; then
#     echo "Singularity container in scratch space is up to date."
#     exit 0
# fi

echo "Syncing singularity container to scratch space from $SINGULARITY_PROJECT_STORAGE_DIR..."
mkdir -p "$SINGULARITY_SCRATCH_STORAGE_DIR"
rsync --update --info=progress3 -ah --temp-dir="$SINGULARITY_SCRATCH_STORAGE_DIR" "$SINGULARITY_CONTAINER_PROJECT" "$SINGULARITY_SCRATCH_STORAGE_DIR"
echo "Container synced to scratch space."