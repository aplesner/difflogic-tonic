#!/bin/bash
#SBATCH --job-name=difflut_sweep
#SBATCH --output=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/difflut_sweep_%A_%a.out
#SBATCH --error=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/difflut_sweep_%A_%a.err
#SBATCH --array=0-17%10
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --nodelist=tikgpu06,tikgpu07,tikgpu08,tikgpu09
#SBATCH --gres=gpu:1

# SLURM Job Array Script for DiffLUT Sweep
# Tests 8 different configurations varying:
# - n (connections per node): 2, 3, 6
# - num_layers: 2, 3
# - default_hidden_size: 256, 512, 1024
# - default_node: neurallut
# Runs at most 4 jobs concurrently

export USERNAME=$(whoami)

# Configuration arrays
N_VALUES=(2 3 6)
NUM_LAYERS=(2 3)
HIDDEN_SIZES=(256 512 1024)
NODE=("neurallut")
# Generate all combinations of parameters
CONFIGS=()
for N in "${N_VALUES[@]}"; do
    for LAYERS in "${NUM_LAYERS[@]}"; do
        for HIDDEN in "${HIDDEN_SIZES[@]}"; do
            for NODE in "${NODE[@]}"; do
                CONFIGS+=("$N,$LAYERS,$HIDDEN,$NODE")
            done
        done
    done
done

# Select configuration based on array task ID (0-indexed)
INDEX=$SLURM_ARRAY_TASK_ID
IFS=',' read -r N LAYERS HIDDEN NODE <<< "${CONFIGS[$INDEX]}"

# Create descriptive job ID
JOB_ID="difflut_n${N}_l${LAYERS}_h${HIDDEN}_${NODE}"

echo "========================================"
echo "SLURM Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment ID: $JOB_ID"
echo "Configuration:"
echo "  n (connections):    $N"
echo "  num_layers:         $LAYERS"
echo "  hidden_size:        $HIDDEN"
echo "  node_type:          $NODE"
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

# Run train.sh with experiment config and overrides
./train.sh experiment=cifar10dvs_difflut \
    base.job_id=$JOB_ID \
    model.difflut.n=$N \
    model.difflut.num_layers=$LAYERS \
    model.difflut.default_hidden_size=$HIDDEN \
    model.difflut.default_node=$NODE \
    train.epochs=5 \
    base.wandb.run_name="${JOB_ID}" \
    '+base.wandb.tags=[sweep]'

# Check return status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully!"
    echo "Configuration: n=$N, layers=$LAYERS, hidden=$HIDDEN, node=$NODE"
    echo "========================================"
    exit 0
else
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID FAILED!"
    echo "Configuration: n=$N, layers=$LAYERS, hidden=$HIDDEN, node=$NODE"
    echo "========================================"
    exit 1
fi
