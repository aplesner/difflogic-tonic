#!/bin/bash
#SBATCH --job-name=train_cifar10dvs
#SBATCH --output=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/train_%A_%a.out
#SBATCH --error=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/train_%A_%a.err
#SBATCH --array=1-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --exclude=tikgpu[06-10]
#SBATCH --gres=gpu:1

# Ensure logs directory exists
export USERNAME=$(whoami)
mkdir -p /itet-stor/${USERNAME}/net_scratch/jobs/difflogic-tonic/


# run these pairs: (0.0,0), (0.0,2), (0.0,4), and (0.05,0) of salt and pepper noise and random crop sizes
VALUES_TO_TEST=(
    "0 0"
    "0 2"
    "0 4"
    "0.05 0"
)

# Select neuron count and tau based on array task ID (1-indexed)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
VALUES_TO_USE=(${VALUES_TO_TEST[$INDEX]})
RANDOM_CROP_SIZE=${VALUES_TO_USE[0]}
SALT_AND_PEPPER_LEVEL=${VALUES_TO_USE[1]}

echo "Selected Random Crop Size: $RANDOM_CROP_SIZE"
echo "Selected Salt and Pepper Level: $SALT_AND_PEPPER_LEVEL"

# Config file
CONFIG_FILE="configs/cifar10dvs_difflogic.yaml"

# Job ID based on random crop size and salt and pepper level
JOB_ID="random_crop_${RANDOM_CROP_SIZE}_salt_pepper_${SALT_AND_PEPPER_LEVEL}_run"

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

# Run train.sh with the selected config and override random crop size and salt and pepper level
./train.sh "$CONFIG_FILE" "$JOB_ID" --override data.augmentation.random_crop_padding=$RANDOM_CROP_SIZE --override data.augmentation.salt_pepper_noise=$SALT_AND_PEPPER_LEVEL --override base.wandb.run_name="c10_dlgn_${RANDOM_CROP_SIZE}crop_${SALT_AND_PEPPER_LEVEL}saltpepper"

# Check return status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully!"
    echo "Random Crop Size: $RANDOM_CROP_SIZE"
    echo "Salt and Pepper Level: $SALT_AND_PEPPER_LEVEL"
    echo "========================================"
    exit 0
else
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID FAILED!"
    echo "Random Crop Size: $RANDOM_CROP_SIZE"
    echo "Salt and Pepper Level: $SALT_AND_PEPPER_LEVEL"
    echo "========================================"
    exit 1
fi
