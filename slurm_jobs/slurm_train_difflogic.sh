#!/bin/bash
#SBATCH --job-name=train_cifar10dvs
#SBATCH --output=/itet-stor/aplesner/net_scratch/jobs/difflogic-tonic/train_%A_%a.out
#SBATCH --error=/itet-stor/aplesner/net_scratch/jobs/difflogic-tonic/train_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --array=1-3
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --exclude=tikgpu[06-10]
#SBATCH --gres=gpu:1

# SLURM Job Array Script for Training
# Tests different neuron counts (8000-64000) on cifar10dvs_difflogic.yaml
# Runs on tikgpu06, tikgpu07, tikgpu09

# Ensure logs directory exists
# mkdir -p /itet-stor/aplesner/net_scratch/jobs/difflogic-tonic/

TAU_VALUES=(1 10 100)

INDEX=$((SLURM_ARRAY_TASK_ID - 1))
TAU=${TAU_VALUES[$INDEX]}

NUM_NEURONS=512000
SALT_PEPPER_NOISE=0.1
RANDOM_CROP=4
DOWNSAMPLE_FACTOR=2

# Config file
CONFIG_FILE="configs/cifar10dvs_difflogic.yaml"

# Job ID based on neuron count
JOB_ID="difflogic_${SLURM_JOB_ID}"

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

WANDB_RUN_NAME="\
c10_dlgn_\
${NUM_NEURONS}neurons_\
${TAU}tau_\
salt_pepper${SALT_PEPPER_NOISE}_\
crop${RANDOM_CROP}\
_downsample${DOWNSAMPLE_FACTOR}"
echo "WandB Run Name: $WANDB_RUN_NAME"


OVERRIDES="\
--override model.difflogic.num_neurons=${NUM_NEURONS} \
--override model.difflogic.tau=${TAU} \
--override base.wandb.run_name=${WANDB_RUN_NAME} \
--override data.downsample_factor=${DOWNSAMPLE_FACTOR} \
--override data.augmentation.salt_pepper_noise=${SALT_PEPPER_NOISE} \
--override data.augmentation.random_crop_padding=${RANDOM_CROP}"

# Run train.sh with the selected config and overrides. The overrides should be enclosed in quotes.
./train.sh "$CONFIG_FILE" "$JOB_ID" "${OVERRIDES}"

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
