#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/jupyter_%j.out
#SBATCH --error=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/jupyter_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1

# SLURM Jupyter Server Launcher
#
# Usage:
#   sbatch slurm_jupyter.sh [PORT]
#
# Default port: 8888
# To use custom port: sbatch slurm_jupyter.sh 8889
#
# After job starts:
# 1. Check the output file for the connection URL
# 2. SSH tunnel: ssh -L PORT:NODE:PORT username@cluster
# 3. Open browser: http://localhost:PORT

set -e

PORT=${1:-8888}

echo "========================================"
echo "Jupyter Server Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Port: $PORT"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"
echo ""

# Go to code directory
cd ~/code/difflogic-tonic || { echo "Error: Could not change to code directory"; exit 1; }

# Source environment variables
source helper_scripts/project_variables.sh

echo "Storage directories:"
echo "  SCRATCH_STORAGE_DIR: $SCRATCH_STORAGE_DIR"
echo "  PROJECT_STORAGE_DIR: $PROJECT_STORAGE_DIR"
echo ""

# Check if container exists
if [ ! -f "$SINGULARITY_CONTAINER" ]; then
    echo "Error: Singularity container not found at $SINGULARITY_CONTAINER"
    exit 1
fi

echo "Using container: $SINGULARITY_CONTAINER"
echo ""

# Get the hostname for connection instructions
HOSTNAME=$(hostname)

echo "========================================"
echo "Starting Jupyter Server"
echo "========================================"
echo ""
echo "Connection Instructions:"
echo "1. Wait for the Jupyter URL to appear below"
echo "2. In a NEW terminal on your LOCAL machine, run:"
echo "   ssh -L ${PORT}:${HOSTNAME}:${PORT} ${USER}@euler.ethz.ch"
echo "3. Open browser and go to: http://localhost:${PORT}"
echo "4. Use the token from the URL below"
echo ""
echo "To stop the server: scancel $SLURM_JOB_ID"
echo ""
echo "========================================"
echo ""

# Start Jupyter in container
# Bind necessary directories
singularity exec \
    --nv \
    --bind ~/code/difflogic-tonic:/workspace \
    --bind $PROJECT_STORAGE_DIR:/project_storage \
    --bind $SCRATCH_STORAGE_DIR:/scratch_storage \
    --pwd /workspace \
    $SINGULARITY_CONTAINER \
    jupyter lab \
        --ip=0.0.0.0 \
        --port=$PORT \
        --no-browser \
        --notebook-dir=/workspace

echo ""
echo "========================================"
echo "Jupyter server stopped"
echo "Job ID: $SLURM_JOB_ID completed"
echo "========================================"
