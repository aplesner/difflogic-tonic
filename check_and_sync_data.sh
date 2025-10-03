#!/bin/bash

# Dataset cache check and sync script
# Usage: ./check_and_sync_data.sh

echo "=== Dataset Cache Check and Sync Script ==="
echo "SCRATCH_STORAGE_DIR: ${SCRATCH_STORAGE_DIR:-not set (using default './scratch/')}"
echo ""

# Add current directory to Python path for src/ imports
export PYTHONPATH="${PYTHONPATH}:."

# Run check and sync
python3 check_and_sync_data.py

# Check return status
if [ $? -ne 0 ]; then
    echo "Check and sync failed!"
    exit 1
fi
