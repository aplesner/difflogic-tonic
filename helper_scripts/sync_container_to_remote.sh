#! /bin/bash

# Check that the PROJECT_NAME is sourced
if [ "$PROJECT_NAME" != "difflogic-tonic" ]; then
    echo "helper_scripts/project_variables.sh is not sourced"
    exit 1
fi

# Sync the singularity container to the remote directory
REMOTE_DIR="${USERNAME}@${REMOTE_SERVER}:${SINGULARITY_PROJECT_STORAGE_DIR}"
echo "Syncing singularity container to remote directory: $REMOTE_DIR"

# Create remote temp dir if it doesn't exist
ssh "${USERNAME}@${REMOTE_SERVER}" "mkdir -p ${SINGULARITY_SCRATCH_STORAGE_DIR}"

# Sync the container to the remote directory
rsync -av --info=progress3 --temp-dir="$SINGULARITY_SCRATCH_STORAGE_DIR" \
    ./singularity/ $REMOTE_DIR
echo "Container synced to remote directory"

