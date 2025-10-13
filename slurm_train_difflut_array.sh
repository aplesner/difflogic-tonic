#!/bin/bash
#SBATCH --job-name=train_cifar10dvs
#SBATCH --output=/itet-stor/davjenny/net_scratch/jobs/difflogic-tonic/train_%A.out
#SBATCH --error=/itet-stor/davjenny/net_scratch/jobs/difflogic-tonic/train_%A.err
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
mkdir -p /itet-stor/davjenny/net_scratch/jobs/difflogic-tonic/

# Neuron counts to test
LAYERS=(
    # 1
    2
    3
    4
)

# Select neuron count based on array task ID (1-indexed)
LAYERS="${LAYERS[$((SLURM_ARRAY_TASK_ID - 1))]}"

# Config file
CONFIG_FILE="configs/cifar10dvs_difflut.yaml"

# Job ID based on neuron count
JOB_ID="layers_${LAYERS}"

echo "========================================"
echo "SLURM Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "Layer Count: $LAYERS"
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
./train.sh "$CONFIG_FILE" "$JOB_ID" --override model.difflut.num_layers=$LAYERS --override base.wandb.run_name="cifar10dvs_difflut_${LAYERS}neurons"

# Check return status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully!"
    echo "Num Layers: $LAYERS"
    echo "========================================"
    exit 0
else
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID FAILED!"
    echo "Num Layers: $LAYERS"
    echo "========================================"
    exit 1
fi
