#!/bin/bash

# Resume training script
# Usage: ./resume.sh <config_file> <job_id>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_file> <job_id>"
    echo "Example: $0 config_nmnist.yaml job_001"
    exit 1
fi

CONFIG_FILE=$1
JOB_ID=$2

echo "Resuming training with config: $CONFIG_FILE, job_id: $JOB_ID"
echo "SCRATCH_STORAGE_DIR: ${SCRATCH_STORAGE_DIR:-not set (using default './scratch/')}"

# Add current directory to Python path for src/ imports
export PYTHONPATH="${PYTHONPATH}:."

python3 main.py $CONFIG_FILE --job_id $JOB_ID --resume