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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import config
from .io_funcs import get_data_splits

logger = logging.getLogger(__name__)


# ============================================================================
# Transform Classes
# ============================================================================

class DownsampleTransform:
    """Apply max pooling downsampling to input tensors"""
    def __init__(self, pool_size: int):
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply max pooling to single sample (C, H, W)"""
        # Convert boolean to float if necessary (max pooling doesn't support bool)
        if x.dtype == torch.bool:
            x = x.float()
        return self.pool(x.unsqueeze(0)).squeeze(0)


class SaltPepperNoise:
    """Add salt and pepper noise by flipping random binary pixels"""
    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Flip random pixels with given probability"""
        if self.probability > 0:
            if x.dtype == torch.bool:
                x = x.float()  # Convert to float for noise addition
            mask = torch.rand_like(x) < self.probability
            # Flip pixels: 0 -> 1, 1 -> 0 (works for binary and float data)
            x = torch.where(mask, 1.0 - x, x)
        return x


class RandomFlip:
    """Apply random horizontal and/or vertical flips"""
    def __init__(self, horizontal: bool = False, vertical: bool = False):
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random flips to single sample (C, H, W)"""
        if self.horizontal and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[2])  # flip width dimension
        if self.vertical and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[1])  # flip height dimension
        return x


class ComposeTransforms:
    """Compose multiple transforms together"""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


class TransformedTensorDataset(Dataset):
    """Wrapper that applies transforms to TensorDataset samples"""
    def __init__(self, tensor_dataset: torch.utils.data.TensorDataset, transform=None):
        self.tensor_dataset = tensor_dataset
        self.transform = transform

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, idx):
        data, label = self.tensor_dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, label


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
    cache_identifier = cfg.data.cache_identifier

    try:
        logger.info(f"Loading cached data for dataset: {dataset_name}")
        if cache_identifier:
            logger.info(f"Using cache variant: {cache_identifier}")
        train_cache, test_cache = get_data_splits(dataset_name, cache_identifier)

        logger.info(f"Cached data loaded successfully.")

    except FileNotFoundError as e:
        raise

    train_dataset = torch.utils.data.TensorDataset(train_cache['data'], train_cache['labels'])
    test_dataset = torch.utils.data.TensorDataset(test_cache['data'], test_cache['labels'])

    # Build transform pipelines
    train_transforms = []
    test_transforms = []

    # Downsampling (applied to both train and test)
    if cfg.data.downsample_pool_size:
        logger.info(f"Adding max pooling downsampling with pool size {cfg.data.downsample_pool_size}")
        downsample = DownsampleTransform(cfg.data.downsample_pool_size)
        train_transforms.append(downsample)
        test_transforms.append(downsample)

    # Augmentation (only for training)
    aug_config = cfg.data.augmentation
    if aug_config.horizontal_flip or aug_config.vertical_flip:
        logger.info(f"Adding random flips: horizontal={aug_config.horizontal_flip}, vertical={aug_config.vertical_flip}")
        train_transforms.append(RandomFlip(
            horizontal=aug_config.horizontal_flip,
            vertical=aug_config.vertical_flip
        ))

    if aug_config.salt_pepper_noise > 0:
        logger.info(f"Adding salt & pepper noise with probability {aug_config.salt_pepper_noise}")
        train_transforms.append(SaltPepperNoise(aug_config.salt_pepper_noise))

    # Wrap datasets with transforms if any
    if train_transforms:
        train_transform = ComposeTransforms(train_transforms)
        train_dataset = TransformedTensorDataset(train_dataset, train_transform)

    if test_transforms:
        test_transform = ComposeTransforms(test_transforms)
        test_dataset = TransformedTensorDataset(test_dataset, test_transform)

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