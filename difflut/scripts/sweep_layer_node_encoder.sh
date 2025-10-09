#!/bin/bash
#SBATCH --job-name=difflut_combo_sweep
#SBATCH --output=/itet-stor/sbuehrer/net_scratch/difflut/logs/combo_sweep_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-215%8
#SBATCH --time=0-04:00:00
#SBATCH --exclude=tikgpu[08,10]

# Sweep over combinations of layer, node and encoder types

# Define absolute paths
SCRATCH_DIR="/itet-stor/sbuehrer/net_scratch/difflut"
EXPERIMENTS_DIR="$SCRATCH_DIR/experiments"

cd $EXPERIMENTS_DIR

# Layer types (same as scripts/sweep_layer_types.sh)
LAYER_TYPES=("random" "learnable" "grouped")

# Node types (same as scripts/sweep_node_types.sh)
NODE_TYPES=(
    "linear_lut"
    "neurallut"
    "polylut"
    "dwn"
    "probabilistic"
    "unbound_probabilistic"
    "hybrid"
    "fourier"
    "fourier_hermitian"
)

# Encoder types (same as scripts/sweep_encoder_types.sh)
ENCODER_TYPES=(
    "thermometer"
    "gaussian_thermometer"
    "distributive_thermometer"
    "gray"
    "onehot"
    "binary"
    "sign_magnitude"
    "logarithmic"
)

total_layers=${#LAYER_TYPES[@]}
total_nodes=${#NODE_TYPES[@]}
total_encoders=${#ENCODER_TYPES[@]}
total=$(( total_layers * total_nodes * total_encoders ))

# SLURM_ARRAY_TASK_ID must be in range 0..total-1
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "SLURM_ARRAY_TASK_ID not set. This script is intended to run as an array job."
    exit 1
fi

if [ $SLURM_ARRAY_TASK_ID -ge $total ] || [ $SLURM_ARRAY_TASK_ID -lt 0 ]; then
    echo "Array task id $SLURM_ARRAY_TASK_ID out of range (0..$((total-1))). Exiting."
    exit 1
fi

# Map the linear index to 3D indices: encoder_idx, layer_idx, node_idx
# We want to iterate encoders first (outer), then layers (middle), then nodes (inner)
i=$SLURM_ARRAY_TASK_ID
per_encoder=$(( total_layers * total_nodes ))
encoder_idx=$(( i / per_encoder ))
rem=$(( i % per_encoder ))
layer_idx=$(( rem / total_nodes ))
node_idx=$(( rem % total_nodes ))

node_type=${NODE_TYPES[$node_idx]}
layer_type=${LAYER_TYPES[$layer_idx]}
encoder_type=${ENCODER_TYPES[$encoder_idx]}

echo "============================================================"
echo "DiffLUT Combo Sweep"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Combination index: node=$node_idx layer=$layer_idx encoder=$encoder_idx"
echo "Testing: node=${node_type}, layer=${layer_type}, encoder=${encoder_type}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "============================================================"

# Run with singularity
singularity exec --nv \
    --bind $SCRATCH_DIR:/workspace \
    $SCRATCH_DIR/containers/pytorch_universal_minimal.sif \
    bash -c "
        export PYTHONPATH=/workspace 
        export CUDA_VISIBLE_DEVICES=0 
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
        cd /workspace/experiments
        python run_experiment.py \
            experiment_name=combo_sweep_${layer_type}_${node_type}_${encoder_type} \
            model.params.layer_type=${layer_type} \
            model.params.node_type=${node_type} \
            model.params.encoder.name=${encoder_type} \
            training.epochs=10 \
            dataloaders.subset_size=10000
    "

exit_code=$?

echo ""
echo "============================================================"
if [ $exit_code -eq 0 ]; then
    echo "Combo ${layer_type}/${node_type}/${encoder_type} completed successfully"
else
    echo "Combo ${layer_type}/${node_type}/${encoder_type} failed with exit code: $exit_code"
fi
echo "============================================================"

exit $exit_code
