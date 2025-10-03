#!/bin/bash

# Test training and resume functionality
# This script tests a few batches of training, saves checkpoint, then resumes

set -e

JOB_ID="test_$(date +%s)"
CONFIG_FILE="config_nmnist.yaml"

# Add current directory to Python path for src/ imports
export PYTHONPATH="${PYTHONPATH}:."

echo "=== Testing Training Pipeline ==="
echo "Job ID: $JOB_ID"
echo "Config: $CONFIG_FILE"
echo "SCRATCH_STORAGE_DIR: ${SCRATCH_STORAGE_DIR:-not set (using default './scratch/')}"
echo ""

# Test 1: Start fresh training for a few batches
echo "=== Test 1: Fresh Training ==="
echo "Starting training for a few batches..."
timeout 60s ./train.sh $CONFIG_FILE $JOB_ID || echo "Training stopped (expected)"
echo ""

# Test 2: Resume training
echo "=== Test 2: Resume Training ==="
echo "Resuming training from checkpoint..."
timeout 60s ./resume.sh $CONFIG_FILE $JOB_ID || echo "Resume training stopped (expected)"
echo ""

echo "=== Test Complete ==="
echo "Check storage directory for checkpoints:"
if [ -n "$SCRATCH_STORAGE_DIR" ]; then
    ls -la $SCRATCH_STORAGE_DIR/storage/
else
    ls -la storage/
fi