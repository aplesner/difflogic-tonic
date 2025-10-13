# Container Updates for Hydra

The Singularity container needs to be updated to include Hydra.

## Required Package

Add to your container definition (`singularity/*.def`):

```
# Hydra for configuration management
hydra-core>=1.3.0
```

## Installation Command

If updating container manually:

```bash
pip install hydra-core>=1.3.0
```

## Verification

Test Hydra installation:

```python
import hydra
from hydra import compose, initialize
print(f"Hydra version: {hydra.__version__}")
```

## Full Requirements (for reference)

Core dependencies needed for the project:

- torch>=2.4.0
- torchvision
- lightning (PyTorch Lightning)
- tonic (event-based vision library)
- omegaconf
- **hydra-core>=1.3.0** (NEW)
- pydantic>=2.0
- wandb
- numpy
- pandas (for future parquet support)
- pyarrow (for future parquet support)

## Rebuild Instructions

After updating the container definition:

```bash
cd singularity/
sudo singularity build difflogic.sif difflogic.def

# Or if using Docker to build:
docker build -t difflogic:latest .
sudo singularity build difflogic.sif docker-daemon://difflogic:latest
```

## Temporary Workaround

If you can't rebuild the container immediately, use the legacy scripts:

```bash
# Legacy training (no Hydra needed)
./legacy/train_old.sh configs/nmnist_difflogic.yaml job_001

# Legacy data prep (no Hydra needed)
./legacy/prepare_data_old.sh configs/prepare_data/nmnist.yaml
```
