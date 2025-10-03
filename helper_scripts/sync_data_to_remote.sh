#!/bin/bash

# Check that the PROJECT_NAME is sourced
if [ "$PROJECT_NAME" != "difflogic-tonic" ]; then
    echo "helper_scripts/project_variables.sh is not sourced"
    exit 1
fi

# Check if local data directory exists
LOCAL_DATA_DIR="./project_storage/data"
if [ ! -d "$LOCAL_DATA_DIR" ]; then
    echo "Local data directory not found: $LOCAL_DATA_DIR"
    echo "Please prepare data first using: python3 prepare_data.py <config_file>"
    exit 1
fi

# Sync local data to the remote data storage directory
REMOTE_DIR="${USER_NAME}@${REMOTE_SERVER}:${DATA_STORAGE_DIR}"

# Sync the data to the remote directory
rsync -av --info=progress2 \
    "$LOCAL_DATA_DIR/" "$REMOTE_DIR"
echo "Data synced to remote directory: $REMOTE_DIR"

