# DiffLogic Tonic

Event-based vision training framework using DiffLogic and baseline models on neuromorphic datasets.

## Quick Start

```bash
# 1. Set environment variables (or add to ~/.bashrc)
source helper_scripts/project_variables.sh

# 2. Prepare dataset (caches processed frames)
./prepare_data.sh prepare=nmnist

# 3. Run training
./train.sh experiment=nmnist_difflogic base.job_id=job_001

# 4. Override parameters
./train.sh experiment=cifar10dvs_difflogic train.epochs=50 model.difflogic.num_neurons=128000
```

**⚡ Now using [Hydra](https://hydra.cc/)** for configuration. See [docs/HYDRA_MIGRATION.md](docs/HYDRA_MIGRATION.md).

## Cluster Setup

### Environment Variables

Set in job scripts or `~/.bashrc`:

```bash
export PROJECT_STORAGE_DIR="/itet-stor/${USER}/net_scratch/projects_storage/difflogic-tonic"
export SCRATCH_STORAGE_DIR="/scratch/${USER}/difflogic-tonic"
export SINGULARITY_CONTAINER="${PROJECT_STORAGE_DIR}/singularity/difflogic.sif"
```

### Storage Locations

```
~/code/difflogic-tonic/                   # Code (git repo)
/itet-stor/${USER}/net_scratch/          # Persistent storage
  └── projects_storage/difflogic-tonic/
      ├── data/                           # Cached datasets
      ├── models/                         # Trained models
      └── singularity/                    # Container
/scratch/${USER}/difflogic-tonic/        # Fast I/O (temporary)
  ├── data/                               # Cached datasets
  └── checkpoints/                        # Training checkpoints
```

**Note**: Container automatically syncs from project storage if not found locally.

## Data Preparation

Convert raw event data to cached frame tensors before training.

```bash
# Single dataset
./prepare_data.sh prepare=nmnist

# With variant override
./prepare_data.sh prepare=cifar10dvs prepare/variants=events_20k

# Custom parameters
./prepare_data.sh prepare=cifar10dvs data.events_per_frame=15000 data.reset_cache=true
```

**Framing modes:**
- **Event count**: Fixed events per frame (e.g., 5k, 10k, 20k)
- **Time window**: Fixed time duration (e.g., 5ms, 10ms, 20ms)

Each variant is cached with identifier: `events_20000_overlap0_denoise5000` 

## Training

```bash
# Use experiment config
./train.sh experiment=nmnist_difflogic base.job_id=job_001

# Mix dataset + model
./train.sh dataset=cifar10dvs model=mlp base.job_id=job_002

# Override parameters
./train.sh experiment=cifar10dvs_difflogic train.epochs=50 train.learning_rate=0.001

# Multiple overrides
./train.sh experiment=nmnist_difflogic \
  model.difflogic.num_neurons=128000 \
  train.epochs=100 \
  data.augmentation.salt_pepper_noise=0.1
```

## Configuration

### Config Structure (Hydra)

```
configs/
├── config.yaml           # Base training defaults
├── prepare_config.yaml   # Base prepare defaults
├── dataset/              # Dataset configs (nmnist, cifar10dvs)
├── model/                # Model configs (difflogic, cnn, mlp)
├── experiment/           # Complete experiments
│   ├── nmnist_difflogic.yaml
│   ├── cifar10dvs_difflogic.yaml
│   └── nmnist_mlp.yaml
└── prepare/              # Data preparation
    ├── nmnist.yaml
    ├── cifar10dvs.yaml
    └── variants/         # Event count/time variants
```

See [docs/HYDRA_MIGRATION.md](docs/HYDRA_MIGRATION.md) for detailed usage.

## Models

Three model types (see [src/model.py](src/model.py)):
1. **DiffLogic** - Differentiable logic-based SNN
2. **CNN** - Convolutional baseline
3. **MLP** - Fully-connected baseline

## SLURM Jobs

```bash
# Interactive Jupyter server
sbatch slurm_jobs/slurm_jupyter.sh [PORT]
# Then: ssh -L PORT:NODE:PORT user@cluster

# Training job arrays
sbatch slurm_jobs/slurm_train_difflogic_array.sh
sbatch slurm_jobs/slurm_prepare_data_array.sh
```

## Helper Scripts

[helper_scripts/](helper_scripts/):
- `sync_code_to_remote.sh` - Sync code to cluster
- `sync_container_to_remote.sh` - Sync container
- `sync_data_to_remote.sh` - Sync datasets
- `sync_logs_from_remote.sh` - Pull logs

## Project Structure

```
configs/                  # YAML configs
src/                      # Source code
  ├── config.py          # Config management
  ├── data.py            # Dataset & transforms
  ├── model.py           # Model definitions
  ├── lightning_module.py # PyTorch Lightning module
  └── prepare_data/      # Data preparation
main.py                   # Training entry
prepare_data.py           # Data prep entry
train.sh                  # Training wrapper
slurm_jobs/              # SLURM scripts
helper_scripts/          # Sync utilities
singularity/             # Container definitions
```

## Requirements

- PyTorch 2.4.0+
- PyTorch Lightning
- CUDA 12.4+
- Tonic (event-based vision library)

See [singularity/](singularity/) for container setup.
