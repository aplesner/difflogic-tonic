#!/bin/bash
#SBATCH --job-name=train_cifar10dvs
#SBATCH --output=/itet-stor/aplesner/net_scratch/jobs/difflogic-tonic/train_%A_%a.out
#SBATCH --error=/itet-stor/aplesner/net_scratch/jobs/difflogic-tonic/train_%A_%a.err
#SBATCH --array=1-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --exclude=tikgpu[06-10]
#SBATCH --gres=gpu:1

# SLURM Job Array Script for Training
# Tests different neuron counts (64000 to 512000) on cifar10dvs_difflogic.yaml
# Runs on all tikgpu nodes except tikgpu06, tikgpu07, tikgpu08, tikgpu09, and tikgpu10

# Ensure logs directory exists
mkdir -p /itet-stor/aplesner/net_scratch/jobs/difflogic-tonic/

# Neuron counts to test
NEURON_COUNTS=(
    64000
    128000
    256000
    512000
)

DOWNSAMPLE_FACTORS=(
    1
    2
    4
    8
)

# Select neuron count and tau based on array task ID (1-indexed)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
NEURON_COUNT="${NEURON_COUNTS[$((INDEX % 4))]}"
DOWNSAMPLE_FACTOR="${DOWNSAMPLE_FACTORS[$((INDEX / 4))]}"

echo "Selected Neuron Count: $NEURON_COUNT"
echo "Selected Downsample Factor: $DOWNSAMPLE_FACTOR"

# Config file
CONFIG_FILE="configs/cifar10dvs_difflogic.yaml"

# Job ID based on neuron count
JOB_ID="neurons_${NEURON_COUNT}_downsample_${DOWNSAMPLE_FACTOR}_run"

echo "========================================"
echo "SLURM Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "Neuron Count: $NEURON_COUNT"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
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
./train.sh "$CONFIG_FILE" "$JOB_ID" --override model.difflogic.num_neurons=$NEURON_COUNT --override data.downsample_pool_size=$DOWNSAMPLE_FACTOR --override base.wandb.run_name="c10_dlgn_${NEURON_COUNT}neurons_${DOWNSAMPLE_FACTOR}downsample"

# Check return status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully!"
    echo "Neuron Count: $NEURON_COUNT"
    echo "Downsample Factor: $DOWNSAMPLE_FACTOR"
    echo "========================================"
    exit 0
else
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID FAILED!"
    echo "Neuron Count: $NEURON_COUNT"
    echo "Downsample Factor: $DOWNSAMPLE_FACTOR"
    echo "========================================"
    exit 1
fi
