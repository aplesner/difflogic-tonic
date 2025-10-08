#!/bin/bash
#SBATCH --job-name=prep_cifar10dvs
#SBATCH --output=/itet-stor/aplesner/net_scratch/jobs/difflogic-tonic/prepare_data_%A_%a.out
#SBATCH --error=/itet-stor/aplesner/net_scratch/jobs/difflogic-tonic/prepare_data_%A_%a.err
#SBATCH --array=1-5
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --nodelist=arton01,arton02,arton03,arton04,arton05,arton06,arton07,arton08,arton09,arton10,arton11

# SLURM Job Array Script for Data Preparation
# Processes 5 different CIFAR10DVS configurations in parallel on arton01 to arton11
# Each job array task processes one configuration variant

# Ensure logs directory exists
mkdir -p /itet-stor/aplesner/net_scratch/jobs/difflogic-tonic/

# Configuration files for each array task
CONFIG_FILES=(
    "configs/prepare_cifar10dvs_events_5k.yaml"
    "configs/prepare_cifar10dvs_events_10k.yaml"
    "configs/prepare_cifar10dvs_events_20k.yaml"
    "configs/prepare_cifar10dvs_events_30k.yaml"
    "configs/prepare_cifar10dvs_events_50k.yaml"
    # "configs/prepare_cifar10dvs_time_5ms.yaml"
    # "configs/prepare_cifar10dvs_time_10ms.yaml"
    # "configs/prepare_cifar10dvs_time_20ms.yaml"
    # "configs/prepare_cifar10dvs_time_50ms.yaml"
    # "configs/prepare_cifar10dvs_time_100ms.yaml"
)

# Select config file based on array task ID (1-indexed)
CONFIG_FILE="${CONFIG_FILES[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "========================================"
echo "SLURM Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
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

# Create storage directories
# mkdir -p "$SCRATCH_STORAGE_DIR"
# mkdir -p "$PROJECT_STORAGE_DIR"

# Run prepare_data.sh with the selected config
./prepare_data.sh "$CONFIG_FILE"

# Check return status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully!"
    echo "Config: $CONFIG_FILE"
    echo "========================================"
    exit 0
else
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID FAILED!"
    echo "Config: $CONFIG_FILE"
    echo "========================================"
    exit 1
fi
