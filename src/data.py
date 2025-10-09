"""Data loading module

This module handles loading pre-prepared TensorDatasets for training.
All heavy preprocessing is done by the prepare_data module.

This keeps training dependencies lightweight and makes the workflow explicit:
1. Run prepare_data.py to cache datasets
2. Run training which loads from cache
"""

import enum
import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


from . import config
from .io_funcs import get_data_splits

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Transform Classes
# ============================================================================

class SaltPepperNoise:
    """Add salt and pepper noise by flipping random binary pixels

    This is a custom transform for binary spike data that isn't available in torchvision.
    Uses v2.ToDtype for dtype conversion when needed.
    """
    def __init__(self, probability: float):
        self.probability = probability
        self.to_float = v2.ToDtype(torch.float32, scale=False)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Flip random pixels with given probability"""
        if self.probability > 0:
            mask = torch.rand(x.shape) < self.probability
            # Flip pixels: 0 -> 1, 1 -> 0 (works for binary and float data)
            x = torch.where(mask, 1.0 - x, x)
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
    # Build transform pipelines using torchvision.transforms.v2
    train_transforms: list[Any] = [
        v2.ToDtype(torch.float32, scale=False)  # Convert bool to float32 for transforms
    ]
    test_transforms: list[Any] = [
        v2.ToDtype(torch.float32, scale=False)  # Convert bool to float32 for transforms
    ]

    # Downsampling (applied to both train and test)
    if cfg.data.downsample_pool_size:
        logger.info(f"Adding max pooling downsampling with pool size {cfg.data.downsample_pool_size}")
        downsample = nn.MaxPool2d(kernel_size=cfg.data.downsample_pool_size, stride=cfg.data.downsample_pool_size)
        train_transforms.append(downsample)
        test_transforms.append(downsample)

    # Augmentation (only for training) - uses torchvision.transforms.v2
    aug_config = cfg.data.augmentation

    # Random crop with padding
    if aug_config.random_crop_padding > 0:
        # Get the spatial dimensions after any downsampling
        sample_data = train_cache['data'][0]
        _, h, w = sample_data.shape
        if cfg.data.downsample_pool_size:
            h = h // cfg.data.downsample_pool_size
            w = w // cfg.data.downsample_pool_size
        logger.info(f"Adding random crop with padding {aug_config.random_crop_padding} (output size: {h}x{w})")
        train_transforms.append(v2.RandomCrop(size=(h, w), padding=aug_config.random_crop_padding))

    # Random flips
    if aug_config.horizontal_flip_probability:
        logger.info(f"Adding random horizontal flip (p={aug_config.horizontal_flip_probability})")
        train_transforms.append(v2.RandomHorizontalFlip(p=aug_config.horizontal_flip_probability))

    if aug_config.vertical_flip_probability:
        logger.info(f"Adding random vertical flip (p={aug_config.vertical_flip_probability})")
        train_transforms.append(v2.RandomVerticalFlip(p=aug_config.vertical_flip_probability))

    # Custom noise augmentation for spike data
    if aug_config.salt_pepper_noise:
        logger.info(f"Adding salt & pepper noise with probability {aug_config.salt_pepper_noise}")
        train_transforms.append(SaltPepperNoise(aug_config.salt_pepper_noise))

    # Additional torchvision transforms can be easily added here:
    # Examples (uncomment and configure as needed):
    # train_transforms.append(v2.RandomRotation(degrees=15))
    # train_transforms.append(v2.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    # train_transforms.append(v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
    # train_transforms.append(v2.RandomErasing(p=0.5, scale=(0.02, 0.33)))
    # train_transforms.append(v2.ColorJitter(brightness=0.2, contrast=0.2))
    # train_transforms.append(v2.RandomPerspective(distortion_scale=0.2, p=0.5))
    # train_transforms.append(v2.ElasticTransform(alpha=50.0))

    # Wrap datasets with transforms if any - uses torchvision.transforms.v2.Compose
    if train_transforms:
        train_transform = v2.Compose(train_transforms)
        train_dataset = TransformedTensorDataset(train_dataset, train_transform)

    if test_transforms:
        test_transform = v2.Compose(test_transforms)
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