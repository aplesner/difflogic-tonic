#!/bin/bash
#SBATCH --job-name=difflut_dataset_sweep
#SBATCH --output=/itet-stor/sbuehrer/net_scratch/difflut/logs/dataset_sweep_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --time=0-04:00:00
#SBATCH --exclude=tikgpu[08,10]

# Sweep over different datasets

# Define absolute paths
SCRATCH_DIR="/itet-stor/sbuehrer/net_scratch/difflut"
EXPERIMENTS_DIR="$SCRATCH_DIR/experiments"

cd $EXPERIMENTS_DIR

# Array of datasets to test
DATASETS=("mnist" "fashionmnist" "cifar10")

# Get dataset for this array task
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "DiffLUT Dataset Sweep"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Testing Dataset: ${dataset}"
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
            experiment_name=dataset_sweep_${dataset} \
            dataloaders=${dataset} \
            training.epochs=10 \
            dataloaders.subset_size=10000
    "

exit_code=$?

echo ""
echo "============================================================"
if [ $exit_code -eq 0 ]; then
    echo "Dataset ${dataset} completed successfully"
else
    echo "Dataset ${dataset} failed with exit code: $exit_code"
fi
echo "============================================================"

exit $exit_code
