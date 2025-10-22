#!/bin/bash
#SBATCH --job-name=difflut_nodes
#SBATCH --output=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/difflut_nodes_%A_%a.out
#SBATCH --error=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/difflut_nodes_%A_%a.err
#SBATCH --array=0-7%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --nodelist=tikgpu06,tikgpu07,tikgpu09
#SBATCH --gres=gpu:1

# SLURM Job Array Script for DiffLUT Node Type Comparison
# Tests all 8 node types with consistent architecture:
# - k=6 inputs per node (n=6)
# - 4 layers
# - 65536 (64k) nodes per layer
# - random layer connections
# Runs at most 4 jobs concurrently

export USERNAME=$(whoami)

# Configuration array: all 8 node types
NODE_TYPES=(
    "linearlut"
    "neurallut"
    "dwn"
    "polylut"
    "probabilistic"
    "hybrid"
    "fourier"
    "gradient_stabilized"
)

# Select node type based on array task ID (0-indexed)
INDEX=$SLURM_ARRAY_TASK_ID
NODE_TYPE="${NODE_TYPES[$INDEX]}"

# Fixed architecture parameters
N=6
NUM_LAYERS=4
HIDDEN_SIZE=65536
LAYER_TYPE="random"

# Create descriptive job ID
JOB_ID="difflut_node_${NODE_TYPE}_n${N}_l${NUM_LAYERS}_h${HIDDEN_SIZE}"

echo "========================================"
echo "SLURM Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment ID: $JOB_ID"
echo "Configuration:"
echo "  node_type:          $NODE_TYPE"
echo "  n (connections):    $N"
echo "  num_layers:         $NUM_LAYERS"
echo "  hidden_size:        $HIDDEN_SIZE"
echo "  layer_type:         $LAYER_TYPE"
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
    model.difflut.num_layers=$NUM_LAYERS \
    model.difflut.default_hidden_size=$HIDDEN_SIZE \
    model.difflut.default_node=$NODE_TYPE \
    model.difflut.default_layer=$LAYER_TYPE \
    train.epochs=5 \
    base.wandb.run_name="${JOB_ID}" \
    '+base.wandb.tags=[node_comparison]'

# Check return status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully!"
    echo "Node type: $NODE_TYPE"
    echo "========================================"
    exit 0
else
    echo ""
    echo "========================================"
    echo "Task $SLURM_ARRAY_TASK_ID FAILED!"
    echo "Node type: $NODE_TYPE"
    echo "========================================"
    exit 1
fi
