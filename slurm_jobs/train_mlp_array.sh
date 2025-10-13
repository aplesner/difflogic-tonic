#!/bin/bash
#SBATCH --job-name=train_cifar10dvs
#SBATCH --output=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/train_%A_%a.out
#SBATCH --error=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/train_%A_%a.err
#SBATCH --array=1-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --nodelist=tikgpu06,tikgpu07,tikgpu09
#SBATCH --gres=gpu:1

# SLURM Job Array Script for Training
# Tests different neuron counts (512-1024) on cifar10dvs_mlp.yaml
# Runs on tikgpu06, tikgpu07, tikgpu09

# Ensure logs directory exists
export USERNAME=$(whoami)
mkdir -p "/itet-stor/${USERNAME}/net_scratch/jobs/difflogic-tonic/"

# Neuron counts to test
NEURON_COUNTS=(
    512
    1024
)

# Select neuron count based on array task ID (1-indexed)
NEURON_COUNT="${NEURON_COUNTS[$((SLURM_ARRAY_TASK_ID - 1))]}"

# Config file
CONFIG_FILE="configs/cifar10dvs_mlp.yaml"

# Job ID based on neuron count
JOB_ID="neurons_${NEURON_COUNT}"

echo "========================================"
echo "SLURM Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "Neuron Count: $NEURON_COUNT"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Node: $(hostname)"
echo "========================================"
echo ""

# Go to code directory
cd ~/code/difflogic-tonic || { echo "Error: Could not change to code directory"; exit 1; }

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Define storage directories
source helper_scripts/project_variables.sh

echo "Storage directories:"
echo "  SCRATCH_STORAGE_DIR: $SCRATCH_STORAGE_DIR"
echo "  PROJECT_STORAGE_DIR: $PROJECT_STORAGE_DIR"
echo ""

# Run train.sh with the selected config and override neuron count
./train.sh "$CONFIG_FILE" "$JOB_ID" --override model.mlp.hidden_size=$NEURON_COUNT --override base.wandb.run_name="cifar10dvs_mlp_${NEURON_COUNT}neurons"

# Check return status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully!"
    echo "Neuron Count: $NEURON_COUNT"
    echo "========================================"
    exit 0
else
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID FAILED!"
    echo "Neuron Count: $NEURON_COUNT"
    echo "========================================"
    exit 1
fi
