# Data Preparation Pipeline - Multiple Framing Variants

## Overview

The data preparation pipeline now supports multiple event framing strategies and can cache different variants of the same dataset with unique identifiers.

## Framing Modes

### 1. Event Count Mode (`frame_mode: event_count`)
Slices events into frames based on a fixed number of events per frame. Good for training CNNs with consistent visual composition.

**Parameters:**
- `events_per_frame`: Number of events to include in each frame
- `overlap`: Number of overlapping events between consecutive frames

### 2. Time Window Mode (`frame_mode: time_window`)
Slices events into frames based on a fixed time duration. Good for temporal consistency in training.

**Parameters:**
- `time_window`: Time duration in microseconds for each frame
- `overlap`: Overlap duration in microseconds between consecutive frames

## Configuration Files

Ten configuration variants have been created for CIFAR10DVS:

### Event Count Variants
1. `configs/prepare_cifar10dvs_events_5k.yaml` - 5,000 events per frame
2. `configs/prepare_cifar10dvs_events_10k.yaml` - 10,000 events per frame
3. `configs/prepare_cifar10dvs_events_20k.yaml` - 20,000 events per frame
4. `configs/prepare_cifar10dvs_events_30k.yaml` - 30,000 events per frame
5. `configs/prepare_cifar10dvs_events_50k.yaml` - 50,000 events per frame

### Time Window Variants
6. `configs/prepare_cifar10dvs_time_5ms.yaml` - 5ms time windows
7. `configs/prepare_cifar10dvs_time_10ms.yaml` - 10ms time windows
8. `configs/prepare_cifar10dvs_time_20ms.yaml` - 20ms time windows
9. `configs/prepare_cifar10dvs_time_50ms.yaml` - 50ms time windows
10. `configs/prepare_cifar10dvs_time_100ms.yaml` - 100ms time windows

## Cache File Naming

Each configuration variant generates uniquely named cache files based on the framing parameters:

**Event Count Mode:**
- `CIFAR10DVS_events_{N}_overlap{M}_denoise{D}_train_data.pt`
- `CIFAR10DVS_events_{N}_overlap{M}_denoise{D}_test_data.pt`

**Time Window Mode:**
- `CIFAR10DVS_time_{T}_overlap{O}_denoise{D}_train_data.pt`
- `CIFAR10DVS_time_{T}_overlap{O}_denoise{D}_test_data.pt`

Examples:
- `CIFAR10DVS_events_20000_overlap0_denoise5000_train_data.pt`
- `CIFAR10DVS_time_10000_overlap0_denoise5000_train_data.pt`

## Usage

### Single Configuration
Run a single configuration using the prepare_data.sh script:

```bash
./prepare_data.sh configs/prepare_cifar10dvs_events_20k.yaml
```

### Parallel Processing with SLURM
Submit all 10 configurations as a job array for parallel processing:

```bash
sbatch slurm_prepare_data_array.sh
```

This will:
- Run 10 parallel jobs (one per configuration)
- Allocate 16 CPUs and 32GB memory per job
- Save logs to `logs/prepare_data_<job_id>_<array_id>.out/err`
- Process all variants simultaneously on the cluster

### Monitor Job Array
```bash
# Check job status
squeue -u $USER

# Check specific job array
squeue -j <job_id>

# View logs
tail -f logs/prepare_data_<job_id>_<array_id>.out
```

## Custom Configuration Options

You can also manually specify a cache identifier in your config:

```yaml
data:
  output_suffix: "custom_identifier"
```

This will use `CIFAR10DVS_custom_identifier_train_data.pt` instead of auto-generating the name.

## Code Changes Summary

1. **[src/prepare_data/config.py](src/prepare_data/config.py)**
   - Added `frame_mode` field (event_count or time_window)
   - Added `time_window` parameter for time-based slicing
   - Added `output_suffix` for custom cache naming
   - Added `get_cache_identifier()` method for automatic naming

2. **[src/prepare_data/prepare.py](src/prepare_data/prepare.py)**
   - Updated ToFrame instantiation to support both modes
   - Passes cache_identifier to save_data_splits()

3. **[src/io_funcs.py](src/io_funcs.py)**
   - Updated `get_data_paths()` to accept cache_identifier
   - Updated `save_data_splits()` to accept cache_identifier
   - Cache files now include framing parameters in filename

4. **[prepare_data.py](prepare_data.py)**
   - Updated logging to show frame mode and parameters
   - Passes cache_identifier throughout

## Environment Variables

- `SCRATCH_STORAGE_DIR`: Fast storage for high I/O operations (recommended)
- `PROJECT_STORAGE_DIR`: Long-term storage for final cached data
- `SINGULARITY_CONTAINER`: Optional path to Singularity container
