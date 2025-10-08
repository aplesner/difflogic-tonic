#!/bin/bash

# Check that the PROJECT_NAME is sourced
if [ "$PROJECT_NAME" != "difflogic-tonic" ]; then
    echo "helper_scripts/project_variables.sh is not sourced"
    exit 1
fi

# Create local directories if they don't exist
LOCAL_DATA_DIR="./project_storage/data"
LOCAL_SINGULARITY_DIR="./singularity"

mkdir -p "$LOCAL_DATA_DIR"
mkdir -p "$LOCAL_SINGULARITY_DIR"

# Define remote directories
REMOTE_DATA_DIR="${USER_NAME}@${REMOTE_SERVER}:${DATA_STORAGE_DIR}"
REMOTE_SINGULARITY_DIR="${USER_NAME}@${REMOTE_SERVER}:${SINGULARITY_STORAGE_DIR}"

echo "Pulling Singularity container from remote..."
rsync -av --info=progress2 \
    "$REMOTE_SINGULARITY_DIR/" "$LOCAL_SINGULARITY_DIR"
echo "Singularity container synced to: $LOCAL_SINGULARITY_DIR"

echo ""
echo "Pulling datasets from remote..."
rsync -av --info=progress2 \
    "$REMOTE_DATA_DIR/" "$LOCAL_DATA_DIR"
echo "Datasets synced to: $LOCAL_DATA_DIR"

echo ""
echo "Pull from remote completed successfully!"
