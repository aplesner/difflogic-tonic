#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/jupyter_%j.out
#SBATCH --error=/itet-stor/%u/net_scratch/jobs/difflogic-tonic/jupyter_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=tikgpu[06-10]

cd ~/code/difflogic-tonic
source helper_scripts/project_variables.sh

# Echo job details
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Allocated GPU: $SLURM_JOB_GPUS"
echo "Running on node: $(hostname)"

# Check things are loaded correctly
echo "Project storage dir: $PROJECT_STORAGE_DIR"
echo "Scratch storage dir: $SCRATCH_STORAGE_DIR"
echo "Singularity container (project): $SINGULARITY_CONTAINER_PROJECT"
echo "Singularity container (scratch): $SINGULARITY_CONTAINER_SCRATCH"

if [ ! -f "$SINGULARITY_CONTAINER_SCRATCH" ]; then
    echo "Copying singularity container to scratch space..."
    mkdir -p "$(dirname $SINGULARITY_CONTAINER_SCRATCH)"   
    rsync -ah "$SINGULARITY_CONTAINER_PROJECT" "$SINGULARITY_CONTAINER_SCRATCH"
fi

singularity exec --nv \
    --bind $PROJECT_STORAGE_DIR,$SCRATCH_STORAGE_DIR \
    $SINGULARITY_CONTAINER_SCRATCH \
    python3 -m notebook --no-browser --port 5998 --ip $(hostname -f)
