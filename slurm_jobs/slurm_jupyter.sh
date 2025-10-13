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

[ ! -f "$SINGULARITY_CONTAINER_SCRATCH" ] && rsync -ah "$SINGULARITY_CONTAINER_PROJECT" "$SINGULARITY_CONTAINER_SCRATCH"

singularity run --nv \
    --bind ~/code/difflogic-tonic \
    --bind $PROJECT_STORAGE_DIR \
    --bind $SCRATCH_STORAGE_DIR \
    $SINGULARITY_CONTAINER_SCRATCH
