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

Prepare and cache event-based datasets before training. The pipeline supports multiple framing strategies for creating dataset variants.

### Single Dataset Variant

```bash
# NMNIST
./prepare_data.sh configs/prepare_nmnist.yaml

# CIFAR10-DVS with specific framing
./prepare_data.sh configs/prepare_cifar10dvs_events_20k.yaml
```

### Multiple Dataset Variants (Parallel)

Use SLURM job arrays to prepare multiple variants in parallel:

```bash
sbatch slurm_prepare_data_array.sh
```

This prepares 10 variants with different framing strategies:
- Event count mode: 5k, 10k, 20k, 30k, 50k events per frame
- Time window mode: 5ms, 10ms, 20ms, 50ms, 100ms time windows

Each variant is cached with a unique identifier (e.g., `events_20000_overlap0_denoise5000`).

### Configuration Options

Configuration options in `configs/prepare_*.yaml`:

**Frame Slicing Mode:**
- `frame_mode`: `event_count` or `time_window`
  - `event_count`: Fixed number of events per frame (consistent visual composition)
  - `time_window`: Fixed time duration per frame (consistent temporal resolution)

**Event Count Mode:**
- `events_per_frame`: Number of events per frame
- `overlap`: Number of overlapping events between frames

**Time Window Mode:**
- `time_window`: Time duration in microseconds per frame
- `overlap`: Overlap duration in microseconds between frames

**Common Options:**
- `denoise_time`: Temporal denoising filter in microseconds (0 to disable)
- `reset_cache`: Force regeneration if `true`
- `output_suffix`: Optional custom cache identifier

**Example:**
```yaml
data:
  frame_mode: event_count
  events_per_frame: 20000
  overlap: 0
  denoise_time: 5000
```

See [PREPARE_DATA_VARIANTS.md](PREPARE_DATA_VARIANTS.md) for detailed documentation.

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
  name: CIFAR10DVS  # NMNIST or CIFAR10DVS

  # Select specific cached variant (optional)
  cache_identifier: "events_20000_overlap0_denoise5000"

  # Input transforms (applied during training)
  downsample_pool_size: 2  # Max pooling: 128x128 -> 64x64

  # Data augmentation (training only)
  augmentation:
    horizontal_flip: true
    vertical_flip: false
    salt_pepper_noise: 0.05  # 5% pixel flip probability

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

### Dataset Variant Selection

Training can use specific cached dataset variants prepared with different framing parameters:

```yaml
data:
  name: CIFAR10DVS
  cache_identifier: "events_20000_overlap0_denoise5000"  # Use 20k events variant
```

Available cache identifiers match the prepare_data output:
- `events_5000_overlap0_denoise5000` - 5k events per frame
- `events_20000_overlap0_denoise5000` - 20k events per frame
- `time_10000_overlap0_denoise5000` - 10ms time window
- etc.

Leave `cache_identifier` as `null` to use the default cache (backward compatible).

See [CACHE_VARIANT_USAGE.md](CACHE_VARIANT_USAGE.md) for complete documentation.

### Input Transforms

Training applies transforms on top of cached data:

**Downsampling:**
- `downsample_pool_size`: Max pooling kernel size (e.g., 2 for 2x2 pooling)
  - Reduces input dimensions: 128×128 → 64×64 (pool_size=2)
  - Applied to both train and test data
  - Model input shape automatically adjusted

**Augmentation** (training only):
- `horizontal_flip`: Random horizontal flip
- `vertical_flip`: Random vertical flip
- `salt_pepper_noise`: Probability to flip random binary pixels (0.0-1.0)

Transforms are applied per-sample before batching, keeping the pipeline flexible and configurable.

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
├── configs/                           # YAML configuration files
│   ├── prepare_*.yaml                # Data preparation configs
│   ├── *_difflogic.yaml              # Training configs
│   └── *_events*.yaml                # Variant-specific training configs
├── src/                               # Source code
│   ├── config.py                     # Config management
│   ├── data.py                       # Dataset loading & transforms
│   ├── model.py                      # Model definitions
│   ├── train.py                      # Training loops
│   ├── io_funcs.py                   # Storage utilities
│   ├── helpers.py                    # Utility functions
│   └── prepare_data/                 # Data preparation module
│       ├── config.py                 # Preparation config
│       └── prepare.py                # Dataset processing
├── main.py                            # Training entry point
├── prepare_data.py                    # Data preparation entry point
├── check_and_sync_data.py             # Data synchronization utility
├── train.sh                           # Training wrapper script
├── prepare_data.sh                    # Data prep wrapper script
├── slurm_prepare_data_array.sh        # SLURM job array for parallel prep
├── singularity/                       # Container definitions
├── helper_scripts/                    # Cluster sync utilities
├── PREPARE_DATA_VARIANTS.md           # Data preparation documentation
└── CACHE_VARIANT_USAGE.md             # Cache variant selection guide
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
