# Using Different Cached Dataset Variants for Training

## Overview

The training pipeline now supports selecting specific cached dataset variants (e.g., different events_per_frame or time_window values) created by the data preparation pipeline.

## How It Works

### 1. Prepare Multiple Dataset Variants

Use the SLURM job array to create multiple cached variants:

```bash
sbatch slurm_prepare_data_array.sh
```

This creates 10 variants with cache identifiers like:
- `events_5000_overlap0_denoise5000`
- `events_10000_overlap0_denoise5000`
- `events_20000_overlap0_denoise5000`
- `events_30000_overlap0_denoise5000`
- `events_50000_overlap0_denoise5000`
- `time_5000_overlap0_denoise5000`
- `time_10000_overlap0_denoise5000`
- `time_20000_overlap0_denoise5000`
- `time_50000_overlap0_denoise5000`
- `time_100000_overlap0_denoise5000`

### 2. Configure Training to Use Specific Variant

In your training config YAML, specify the `cache_identifier`:

```yaml
data:
  name: CIFAR10DVS
  cache_identifier: "events_20000_overlap0_denoise5000"  # Use 20k events variant

  # Transforms are applied on top of cached data
  downsample_pool_size: 2
  augmentation:
    horizontal_flip: true
    salt_pepper_noise: 0.05
```

### 3. Run Training

```bash
python main.py configs/cifar10dvs_difflogic_events20k.yaml
```

## Cache File Naming Convention

Cache files follow this pattern:
```
{dataset_name}_{cache_identifier}_train_data.pt
{dataset_name}_{cache_identifier}_test_data.pt
```

Examples:
- `CIFAR10DVS_events_20000_overlap0_denoise5000_train_data.pt`
- `CIFAR10DVS_time_10000_overlap0_denoise5000_train_data.pt`

## Configuration Options

### Complete Example

```yaml
data:
  name: CIFAR10DVS

  # Select specific cached variant (null = default/backward compatible)
  cache_identifier: "events_20000_overlap0_denoise5000"

  # Input transforms (applied during training)
  downsample_pool_size: 2  # Max pooling 2x2

  # Augmentation (training only)
  augmentation:
    horizontal_flip: true
    vertical_flip: false
    salt_pepper_noise: 0.05
```

## Comparison Workflow

To compare different event framing strategies:

1. **Prepare all variants** (one-time):
   ```bash
   sbatch slurm_prepare_data_array.sh
   ```

2. **Create config for each variant**:
   - `configs/cifar10dvs_events_5k.yaml` with `cache_identifier: "events_5000_overlap0_denoise5000"`
   - `configs/cifar10dvs_events_10k.yaml` with `cache_identifier: "events_10000_overlap0_denoise5000"`
   - `configs/cifar10dvs_events_20k.yaml` with `cache_identifier: "events_20000_overlap0_denoise5000"`
   - etc.

3. **Run training jobs**:
   ```bash
   python main.py configs/cifar10dvs_events_5k.yaml --job_id run_5k
   python main.py configs/cifar10dvs_events_10k.yaml --job_id run_10k
   python main.py configs/cifar10dvs_events_20k.yaml --job_id run_20k
   ```

4. **Compare results** using job_id tracking

## Backward Compatibility

If `cache_identifier` is `null` or omitted, the system uses the default cache naming:
- `{dataset_name}_train_data.pt`
- `{dataset_name}_test_data.pt`

This maintains compatibility with existing cached datasets.

## Implementation Details

### Key Files Modified

1. **[src/config.py](src/config.py:44)** - Added `cache_identifier` field to `DataConfig`
2. **[src/io_funcs.py](src/io_funcs.py:101)** - Updated `get_data_splits()` to accept cache_identifier
3. **[src/data.py](src/data.py:133)** - Passes cache_identifier from config to data loading

### Cache Discovery

To see all available cached variants:

```python
from src.io_funcs import discover_datasets

datasets = discover_datasets()
for name, paths in datasets.items():
    print(f"{name}:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
```

This will show all cached variants in both scratch and project storage.

## Example Configs

- **Default**: [configs/cifar10dvs_difflogic.yaml](configs/cifar10dvs_difflogic.yaml) - No cache_identifier (uses default)
- **20k Events**: [configs/cifar10dvs_difflogic_events20k.yaml](configs/cifar10dvs_difflogic_events20k.yaml) - Uses 20k events variant

## Error Handling

If the specified cache_identifier doesn't exist, you'll get a clear error message:

```
FileNotFoundError: Cached data not found for dataset 'CIFAR10DVS' with identifier 'events_20000_overlap0_denoise5000'
Checked locations:
  Scratch: /path/to/scratch/data/CIFAR10DVS_events_20000_overlap0_denoise5000_train_data.pt (missing)
  Project: /path/to/project/data/CIFAR10DVS_events_20000_overlap0_denoise5000_train_data.pt (missing)

Please run 'python3 prepare_data.py <config_file>' to prepare the dataset first.
```
