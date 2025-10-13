#!/bin/bash
#SBATCH --job-name=difflut_exp
#SBATCH --output=/itet-stor/sbuehrer/net_scratch/difflut/logs/experiment_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --exclude=tikgpu[08,10]

# Single experiment run with configurable parameters via Hydra
# Usage: sbatch run_experiment.sh [hydra overrides]
# Example: sbatch run_experiment.sh dataset=fashionmnist training.epochs=20

# Define absolute paths
SCRATCH_DIR="/itet-stor/sbuehrer/net_scratch/difflut"
EXPERIMENTS_DIR="$SCRATCH_DIR/experiments"

cd $EXPERIMENTS_DIR

echo "============================================================"
echo "DiffLUT Experiment Runner"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "============================================================"

# Get Hydra overrides from command line arguments
HYDRA_ARGS="$@"

if [ -n "$HYDRA_ARGS" ]; then
    echo "Hydra overrides: $HYDRA_ARGS"
else
    echo "Using default configuration"
fi

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
        python run_experiment.py $HYDRA_ARGS
    "

exit_code=$?

echo ""
echo "============================================================"
if [ $exit_code -eq 0 ]; then
    echo "Experiment completed successfully"
else
    echo "Experiment failed with exit code: $exit_code"
fi
echo "============================================================"

exit $exit_code
