"""Data loading module

This module handles loading pre-prepared TensorDatasets for training.
All heavy preprocessing is done by the prepare_data module.

This keeps training dependencies lightweight and makes the workflow explicit:
1. Run prepare_data.py to cache datasets
2. Run training which loads from cache
"""

import enum
import logging

import torch
from torch.utils.data import DataLoader

from . import config
from .io_funcs import get_data_splits

logger = logging.getLogger(__name__)


class Datasets(enum.Enum):
    NMNIST = "NMNIST"
    CIFAR10DVS = "CIFAR10DVS"


class SensorSizes(enum.Enum):
    """Sensor sizes in (Height, Width, Channels) format - NHWC

    Note: Data is stored in NCHW format, but these constants represent
    the original sensor dimensions in HWC order for reference.
    """
    NMNIST = (34, 34, 2)  # tonic.datasets.NMNIST.sensor_size
    CIFAR10DVS = (128, 128, 2)  # tonic.datasets.CIFAR10DVS.sensor_size


class OutputClasses(enum.Enum):
    """Number of output classes for each dataset"""
    NMNIST = 10
    CIFAR10DVS = 10


def get_dataloaders(cfg: config.Config) -> tuple[DataLoader, DataLoader]:
    """Get dataloaders from pre-prepared cached data

    This function ONLY loads pre-prepared datasets. It does NOT prepare data.
    Run prepare_data.py first to cache datasets.

    Cache Priority: Scratch -> Project -> Error

    Args:
        cfg: Complete configuration object

    Returns:
        Tuple of (train_dataloader, test_dataloader)

    Raises:
        FileNotFoundError: If cached data is not found in either storage location
    """
    dataset_name = cfg.data.name

    try:
        logger.info(f"Loading cached data for dataset: {dataset_name}")
        train_cache, test_cache = get_data_splits(dataset_name)
        
        logger.info(f"Cached data loaded successfully.")

    except FileNotFoundError as e:
        raise

    train_dataset = torch.utils.data.TensorDataset(train_cache['data'], train_cache['labels'])
    test_dataset = torch.utils.data.TensorDataset(test_cache['data'], test_cache['labels'])

    # Create dataloaders with training config
    pin_memory_device = "cuda" if torch.cuda.is_available() else "cpu"
    train_config = cfg.train
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=train_config.dataloader.batch_size,
        prefetch_factor=train_config.dataloader.prefetch_factor,
        shuffle=train_config.dataloader.shuffle_train,
        num_workers=train_config.dataloader.num_workers,
        pin_memory=train_config.dataloader.pin_memory if torch.cuda.is_available() else False,
        pin_memory_device=pin_memory_device if train_config.dataloader.pin_memory else ""
    )

    dataloader_test = DataLoader(
        test_dataset,
        batch_size=train_config.dataloader.batch_size,
        prefetch_factor=train_config.dataloader.prefetch_factor,
        shuffle=False,
        num_workers=train_config.dataloader.num_workers,
        pin_memory=train_config.dataloader.pin_memory if torch.cuda.is_available() else False,
        pin_memory_device=pin_memory_device if train_config.dataloader.pin_memory else ""
    )

    logger.info(f"Dataloaders created with {len(train_dataset)} train and {len(test_dataset)} test samples")

    return dataloader_train, dataloader_test