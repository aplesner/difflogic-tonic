# DiffLogic Tonic

Event-based vision training framework for difflogic on neuromorphic datasets.

## Quick Start

```bash
# 1. Prepare dataset
./prepare_data.sh configs/prepare_nmnist.yaml

# 2. Run training
./train.sh configs/nmnist_difflogic.yaml job_001

# 3. Resume training (optional)
./train.sh configs/nmnist_difflogic.yaml job_001 --resume
```

## Cluster Setup

### Environment Variables

Set these in your job scripts or `~/.bashrc`:

```bash
export PROJECT_STORAGE_DIR="/itet-stor/${USER}/net_scratch/projects_storage/difflogic-tonic"
export SCRATCH_STORAGE_DIR="/scratch/${USER}/difflogic-tonic"
export SINGULARITY_CONTAINER="${PROJECT_STORAGE_DIR}/singularity/difflogic.sif"
```

See [helper_scripts/project_variables.sh](helper_scripts/project_variables.sh) for reference.

### File Locations

```
/home/${USER}/code/difflogic-tonic/          # Code (this repo)
/itet-stor/${USER}/net_scratch/              # Project storage (persistent)
├── projects_storage/difflogic-tonic/
    ├── data/                                 # Cached datasets (long-term)
    ├── models/                               # Trained models
    └── singularity/                          # Container images
/scratch/${USER}/difflogic-tonic/            # Scratch storage (high I/O)
├── data/                                     # Cached datasets (fast access)
└── checkpoints/                              # Training checkpoints
```

## Storage Strategy

- **Scratch** (`/scratch/`): High I/O throughput, temporary, cleared periodically
- **Project** (`/itet-stor/`): Persistent storage, lower I/O performance

Datasets are prepared to both locations. Training automatically uses scratch for speed and falls back to project storage if needed.

## Data Preparation

Prepare and cache event-based datasets before training:

```bash
# NMNIST
./prepare_data.sh configs/prepare_nmnist.yaml

# CIFAR10-DVS
./prepare_data.sh configs/prepare_cifar10dvs.yaml
```

Configuration options in `configs/prepare_*.yaml`:
- `events_per_frame`: Events per input frame
- `overlap`: Overlap between consecutive frames
- `denoise_time`: Temporal denoising filter (microseconds)
- `reset_cache`: Force regeneration if `true`

### Check Data Status

```bash
./check_and_sync_data.sh
```

Discovers datasets and syncs between scratch/project storage if needed.


### Upload to the cluster 

```bash
./helper_scripts/sync_data_to_remote.sh
```
Uploads the datasets to the cluster 

## Training

### Basic Training

```bash
./train.sh configs/nmnist_difflogic.yaml my_job_id
```

Remember to set the singularity environment variable. The easiest is to run `source helper_scripts/project_variables.sh`

### Resume from Checkpoint

```bash
./train.sh configs/nmnist_difflogic.yaml my_job_id --resume
```

### Override Config Values

```bash
./train.sh configs/nmnist_difflogic.yaml my_job_id \
  --override "train.epochs=10 train.learning_rate=0.001"
```

### Debug Mode

```bash
./train.sh configs/nmnist_difflogic.yaml my_job_id --debug
```

## Configuration Files

Configs are in [configs/](configs/):

- `prepare_nmnist.yaml`, `prepare_cifar10dvs.yaml` - Data preparation
- `nmnist_difflogic.yaml` - DiffLogic model on NMNIST
- `nmnist_cnn.yaml` - CNN baseline
- `nmnist_mlp.yaml` - MLP baseline

### Key Config Sections

```yaml
base:
  seed: 42
  job_id: "my_experiment"

data:
  name: NMNIST  # NMNIST or CIFAR10DVS

train:
  epochs: 3
  learning_rate: 1e-2
  device: cuda
  checkpoint_interval_minutes: 10.0
  dataloader:
    batch_size: 64
    num_workers: 6

model:
  model_type: DiffLogic  # DiffLogic, CNN, or MLP
  difflogic:
    num_neurons: 4000
    num_layers: 4
```

## Models

Three model types supported (see [src/model.py](src/model.py)):

1. **DiffLogic** - Differentiable logic-based SNN
2. **CNN** - Convolutional baseline
3. **MLP** - Fully-connected baseline

## Helper Scripts

Located in [helper_scripts/](helper_scripts/):

- `sync_code_to_remote.sh` - Sync code to cluster
- `sync_container_to_remote.sh` - Sync Singularity container
- `sync_data_to_remote.sh` - Sync prepared datasets
- `sync_logs_from_remote.sh` - Pull logs from cluster

## Project Structure

```
.
├── configs/                # YAML configuration files
├── src/                    # Source code
│   ├── config.py          # Config management
│   ├── data.py            # Dataset loading
│   ├── model.py           # Model definitions
│   ├── train.py           # Training loops
│   ├── io_funcs.py        # Storage utilities
│   └── helpers.py         # Utility functions
├── main.py                # Training entry point
├── prepare_data.py        # Data preparation entry point
├── check_and_sync_data.py # Data synchronization utility
├── train.sh               # Training wrapper script
├── prepare_data.sh        # Data prep wrapper script
├── singularity/           # Container definitions
└── helper_scripts/        # Cluster sync utilities
```

## Checkpointing

Checkpoints are saved to `${SCRATCH_STORAGE_DIR}/checkpoints/` every `checkpoint_interval_minutes` (default: 10 minutes).

Use `--resume` to continue training from the last checkpoint.

## Requirements

Training requires:
- PyTorch 2.4.0+
- CUDA 12.4+
- Tonic (event-based vision library)

See [singularity/](singularity/) for container definitions.
