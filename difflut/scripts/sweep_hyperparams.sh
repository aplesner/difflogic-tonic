#!/bin/bash
#SBATCH --job-name=difflut_hp_sweep
#SBATCH --output=/itet-stor/sbuehrer/net_scratch/difflut/logs/hyperparam_sweep_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --time=0-04:00:00
#SBATCH --exclude=tikgpu[08,10]

# Hyperparameter sweep: learning rate x hidden size

# Define absolute paths
SCRATCH_DIR="/itet-stor/sbuehrer/net_scratch/difflut"
EXPERIMENTS_DIR="$SCRATCH_DIR/experiments"

cd $EXPERIMENTS_DIR

# Learning rates to test
LRS=(0.001 0.01 0.1)
# Hidden sizes to test
HIDDEN_SIZES=(500 1000 2000)

# Calculate grid indices
lr_idx=$((SLURM_ARRAY_TASK_ID / 3))
hs_idx=$((SLURM_ARRAY_TASK_ID % 3))

lr=${LRS[$lr_idx]}
hidden_size=${HIDDEN_SIZES[$hs_idx]}

echo "============================================================"
echo "DiffLUT Hyperparameter Sweep"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Learning Rate: ${lr}"
echo "Hidden Size: ${hidden_size}"
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
            experiment_name=hp_sweep_lr${lr}_hs${hidden_size} \
            training.lr=${lr} \
            'model.params.hidden_sizes=[${hidden_size},${hidden_size}]' \
            training.epochs=10 \
            dataloaders.subset_size=10000
    "

exit_code=$?

echo ""
echo "============================================================"
if [ $exit_code -eq 0 ]; then
    echo "HP sweep (lr=${lr}, hs=${hidden_size}) completed successfully"
else
    echo "HP sweep (lr=${lr}, hs=${hidden_size}) failed with exit code: $exit_code"
fi
echo "============================================================"

exit $exit_code
