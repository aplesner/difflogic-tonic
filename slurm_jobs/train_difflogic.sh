#!/bin/bash
#SBATCH --job-name=train_cifar10dvs
#SBATCH --output=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/train_%A_%a.out
#SBATCH --error=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/train_%A_%a.err
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
export USERNAME=$(whoami)
mkdir -p "/itet-stor/${USERNAME}/net_scratch/jobs/difflogic-tonic/"

TAU_VALUES=(1 10 100)

INDEX=$((SLURM_ARRAY_TASK_ID - 1))
TAU=${TAU_VALUES[$INDEX]}

NUM_NEURONS=512000
SALT_PEPPER_NOISE=0.1
RANDOM_CROP=4
DOWNSAMPLE_FACTOR=2

# Job ID based on job number
JOB_ID="difflogic_${SLURM_JOB_ID}"

echo "========================================"
echo "SLURM Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: cifar10dvs_difflogic"
echo "Neuron Count: $NUM_NEURONS"
echo "Tau: $TAU"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Node: $(hostname)"
echo "========================================"
echo ""

# Go to code directory
cd ~/code/difflogic-tonic || { echo "Error: Could not change to code directory"; exit 1; }

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

# Run train.sh with experiment config and overrides
./train.sh experiment=cifar10dvs_difflogic base.job_id=$JOB_ID model.difflogic.num_neurons=${NUM_NEURONS} model.difflogic.tau=${TAU} base.wandb.run_name=${WANDB_RUN_NAME} data.downsample_pool_size=${DOWNSAMPLE_FACTOR} data.augmentation.salt_pepper_noise=${SALT_PEPPER_NOISE} data.augmentation.random_crop_padding=${RANDOM_CROP}

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
