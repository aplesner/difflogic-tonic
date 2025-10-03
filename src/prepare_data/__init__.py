"""Data preparation module

This module handles the heavy lifting of dataset preparation:
- Loading raw event-based datasets (NMNIST, CIFAR10DVS)
- Preprocessing with tonic transforms
- Caching processed TensorDatasets to both scratch and project storage

Separation from training code keeps dependencies clean and makes
the preparation step explicit and reusable.
"""

from .config import PrepareDataConfig
from .prepare import prepare_dataset

__all__ = ['PrepareDataConfig', 'prepare_dataset']