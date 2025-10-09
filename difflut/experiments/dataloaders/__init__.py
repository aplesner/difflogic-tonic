"""
Data loaders for DiffLUT experiments.
"""

from dataloaders.base_dataloader import get_dataloader, register_dataloader
from dataloaders.image_dataloader import MNISTDataLoader, FashionMNISTDataLoader, CIFAR10DataLoader

__all__ = [
    'get_dataloader',
    'register_dataloader',
    'MNISTDataLoader',
    'FashionMNISTDataLoader',
    'CIFAR10DataLoader',
]
