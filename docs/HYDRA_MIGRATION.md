# Hydra Configuration Migration Guide

The project now uses [Hydra](https://hydra.cc/) for configuration management, providing cleaner composition and CLI overrides.

## What Changed

### Old System (OmegaConf directly)
```bash
./train.sh configs/nmnist_difflogic.yaml job_001 --override "train.epochs=10"
```

### New System (Hydra)
```bash
./train.sh experiment=nmnist_difflogic base.job_id=job_001 train.epochs=10
```

## Benefits

1. **Config Composition**: Base configs + dataset + model + experiment overrides
2. **Cleaner CLI**: No need for `--override` flag, direct key=value syntax
3. **Less Repetition**: Share common settings across configs
4. **WandB Sweeps**: Easy integration with Hydra's sweep functionality
5. **Type Safety**: Still uses Pydantic for validation after Hydra loads

## Configuration Structure

```
configs/
├── config.yaml              # Base training config (defaults)
├── prepare_config.yaml      # Base prepare_data config (defaults)
├── dataset/                 # Dataset-specific overrides
│   ├── nmnist.yaml
│   └── cifar10dvs.yaml
├── model/                   # Model-specific overrides
│   ├── difflogic.yaml
│   ├── cnn.yaml
│   └── mlp.yaml
├── experiment/              # Complete experiment configs
│   ├── nmnist_difflogic.yaml
│   ├── cifar10dvs_difflogic.yaml
│   └── nmnist_mlp.yaml
└── prepare/                 # Prepare data configs
    ├── nmnist.yaml
    ├── cifar10dvs.yaml
    └── variants/
        ├── events_5k.yaml
        ├── events_10k.yaml
        ├── events_20k.yaml
        └── time_10ms.yaml
```

## Usage Examples

### Training

```bash
# Use experiment config (includes dataset + model)
./train.sh experiment=nmnist_difflogic

# Mix dataset + model configs
./train.sh dataset=cifar10dvs model=mlp

# Override specific values
./train.sh experiment=cifar10dvs_difflogic train.epochs=50 train.learning_rate=0.001

# Override nested values
./train.sh experiment=nmnist_difflogic model.difflogic.num_neurons=128000

# Multiple overrides
./train.sh experiment=cifar10dvs_difflogic \
  base.job_id=my_exp \
  train.epochs=100 \
  model.difflogic.num_neurons=64000 \
  data.augmentation.salt_pepper_noise=0.1
```

### Data Preparation

```bash
# Use prepare config
./prepare_data.sh prepare=nmnist

# Use variant override
./prepare_data.sh prepare=cifar10dvs prepare/variants=events_20k

# Override specific values
./prepare_data.sh prepare=cifar10dvs data.events_per_frame=15000

# Force cache reset
./prepare_data.sh prepare=nmnist data.reset_cache=true
```

## Config Composition Order

Hydra loads configs in this order (later overrides earlier):

1. Base config (`config.yaml` or `prepare_config.yaml`)
2. Default group configs (e.g., `dataset/nmnist.yaml`, `model/difflogic.yaml`)
3. Experiment config (e.g., `experiment/cifar10dvs_difflogic.yaml`)
4. CLI overrides (e.g., `train.epochs=50`)

## Creating New Configs

### New Experiment

Create `configs/experiment/my_experiment.yaml`:

```yaml
# @package _global_

defaults:
  - override /dataset: cifar10dvs
  - override /model: difflogic

base:
  job_id: "my_experiment"
  wandb:
    online: true
    tags: ["my_tag"]

train:
  epochs: 100
  learning_rate: 0.005

model:
  difflogic:
    num_neurons: 128000
```

Then run: `./train.sh experiment=my_experiment`

### New Data Preparation Variant

Create `configs/prepare/variants/events_15k.yaml`:

```yaml
# @package _global_

data:
  frame_mode: event_count
  events_per_frame: 15000
  overlap: 0.0
```

Then run: `./prepare_data.sh prepare=cifar10dvs prepare/variants=events_15k`

## Legacy Configs

Old YAML configs are still available in the root `configs/` directory but are not used by Hydra.
Legacy scripts are in `legacy/` folder for reference.

## Debugging

To see the final composed config:

```bash
# Training
./train.sh experiment=nmnist_difflogic --cfg job

# Data prep
./prepare_data.sh prepare=nmnist --cfg job
```

## Migration Checklist

- [x] Create Hydra config structure
- [x] Convert main.py to use Hydra
- [x] Convert prepare_data.py to use Hydra
- [x] Update train.sh for Hydra CLI
- [x] Update prepare_data.sh for Hydra CLI
- [ ] Update SLURM job scripts to use Hydra syntax
- [ ] Update README with Hydra examples
- [ ] Update container with hydra-core dependency
