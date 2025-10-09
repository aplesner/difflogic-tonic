#!/bin/bash
#SBATCH --job-name=difflut_encoder_sweep
#SBATCH --output=/itet-stor/sbuehrer/net_scratch/difflut/logs/encoder_sweep_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --time=0-04:00:00
#SBATCH --exclude=tikgpu[08,10]

# Sweep over different encoder types

# Define absolute paths
SCRATCH_DIR="/itet-stor/sbuehrer/net_scratch/difflut"
EXPERIMENTS_DIR="$SCRATCH_DIR/experiments"

cd $EXPERIMENTS_DIR

# Array of encoder types to test (must match difflut/encoder implementations and the config name values)
# Names correspond to the `model.params.encoder.name` config key used by the project
ENCODER_TYPES=(
    "thermometer"
    "gaussian_thermometer"
    "distributive_thermometer"
    "gray"
    "one_hot"
    "binary"
    "sign_magnitude"
    "logarithmic"
)

# Get encoder type for this array task
encoder_type=${ENCODER_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "DiffLUT Encoder Type Sweep"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Testing Encoder Type: ${encoder_type}"
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
            experiment_name=encoder_sweep_${encoder_type} \
            model.params.encoder.name=${encoder_type} \
            training.epochs=10 \
            dataloaders.subset_size=10000
    "

exit_code=$?

echo ""
echo "============================================================"
if [ $exit_code -eq 0 ]; then
    echo "Encoder type ${encoder_type} completed successfully"
else
    echo "Encoder type ${encoder_type} failed with exit code: $exit_code"
fi
echo "============================================================"

exit $exit_code
