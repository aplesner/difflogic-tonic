#!/bin/bash
#SBATCH --job-name=difflut_layer_sweep
#SBATCH --output=/itet-stor/sbuehrer/net_scratch/difflut/logs/layer_sweep_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --time=0-04:00:00
#SBATCH --exclude=tikgpu[08,10]

# Sweep over different layer types

# Define absolute paths
SCRATCH_DIR="/itet-stor/sbuehrer/net_scratch/difflut"
EXPERIMENTS_DIR="$SCRATCH_DIR/experiments"

cd $EXPERIMENTS_DIR

# Array of layer types to test (must match difflut/layers implementations)
LAYER_TYPES=("random" "learnable" "adaptive")

# Get layer type for this array task
layer_type=${LAYER_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "DiffLUT Layer Type Sweep"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Testing Layer Type: ${layer_type}"
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
            experiment_name=layer_sweep_${layer_type} \
            model.params.layer_type=${layer_type} \
            training.epochs=10 \
            dataloaders.subset_size=10000
    "

exit_code=$?

echo ""
echo "============================================================"
if [ $exit_code -eq 0 ]; then
    echo "Layer type ${layer_type} completed successfully"
else
    echo "Layer type ${layer_type} failed with exit code: $exit_code"
fi
echo "============================================================"

exit $exit_code
