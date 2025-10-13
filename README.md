# DiffLogic Tonic

Event-based vision training framework using DiffLogic and baseline models on neuromorphic datasets.

## Quick Start

```bash
# Set environment variables
source helper_scripts/project_variables.sh

# Prepare dataset
./prepare_data.sh prepare=nmnist

# Train model
./train.sh experiment=nmnist_difflogic base.job_id=job_001

# Override parameters
./train.sh experiment=cifar10dvs_difflogic train.epochs=50 model.difflogic.num_neurons=128000
```

Configuration uses [Hydra](https://hydra.cc/) with composition from `configs/`.

## Configuration

Structure:
```
configs/
├── config.yaml          # Base training defaults
├── prepare_config.yaml  # Base prepare defaults
├── dataset/             # nmnist, cifar10dvs
├── model/               # difflogic, cnn, mlp
├── experiment/          # Complete experiments
└── prepare/             # Data preparation
    └── variants/        # Event count/time variants
```

Examples:
```bash
# Training
./train.sh experiment=nmnist_difflogic
./train.sh dataset=cifar10dvs model=mlp train.epochs=100

# Data preparation
./prepare_data.sh prepare=nmnist
./prepare_data.sh prepare=cifar10dvs prepare/variants=events_20k
```

## Models

- **DiffLogic** - Differentiable logic-based SNN
- **CNN** - Convolutional baseline
- **MLP** - Fully-connected baseline

## SLURM

```bash
sbatch slurm_jobs/slurm_jupyter.sh
sbatch slurm_jobs/slurm_train_difflogic_array.sh
sbatch slurm_jobs/slurm_prepare_data_array.sh
```

## Requirements

- PyTorch 2.4.0+, PyTorch Lightning
- Tonic, Hydra
- CUDA 12.4+

Container: see `singularity/`
