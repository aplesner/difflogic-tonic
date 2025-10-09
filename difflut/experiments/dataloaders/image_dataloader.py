#!/usr/bin/env python3
"""
Image dataset loaders for MNIST, FashionMNIST, and CIFAR10.
Handles data loading and preprocessing.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional
import numpy as np

from dataloaders.base_dataloader import BaseDataLoader, register_dataloader


@register_dataloader("mnist")
class MNISTDataLoader(BaseDataLoader):
    """DataLoader for MNIST dataset."""
    
    def __init__(self, config: dict):
        """
        Initialize MNIST dataloader.
        
        Args:
            config: Dataset configuration dictionary
        """
        super().__init__(config)
        self.data_dir = config.get('data_dir', './data')
        self.batch_size = config.get('batch_size', 128)
        self.subset_size = config.get('subset_size', None)
        self.test_subset_size = config.get('test_subset_size', 1000)
        self.num_workers = config.get('num_workers', 0)
        
        self.input_size = 784  # MNIST flattened size
    
    def prepare_data(self):
        """Download MNIST dataset if needed."""
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)
    
    def setup(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Setup train, validation, and test dataloaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Define transforms - flatten images
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        
        # Load datasets
        train_dataset = datasets.MNIST(
            self.data_dir, train=True, download=False, transform=transform
        )
        test_dataset = datasets.MNIST(
            self.data_dir, train=False, download=False, transform=transform
        )
        
        # Create subsets if specified
        if self.subset_size:
            train_indices = torch.randperm(len(train_dataset))[:self.subset_size]
            train_dataset = Subset(train_dataset, train_indices)
        
        if self.test_subset_size:
            test_indices = torch.randperm(len(test_dataset))[:self.test_subset_size]
            test_dataset = Subset(test_dataset, test_indices)
        
        # Split train into train and validation (90/10 split)
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create dataloaders (no encoding here - model will handle it)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def get_input_size(self) -> int:
        """Get the raw input size (before encoding)."""
        return self.input_size
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return 10


@register_dataloader("fashionmnist")
class FashionMNISTDataLoader(BaseDataLoader):
    """DataLoader for FashionMNIST dataset."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.data_dir = config.get('data_dir', './data')
        self.batch_size = config.get('batch_size', 128)
        self.subset_size = config.get('subset_size', None)
        self.test_subset_size = config.get('test_subset_size', 1000)
        self.num_workers = config.get('num_workers', 0)
        
        self.input_size = 784  # FashionMNIST flattened size
    
    def prepare_data(self):
        """Download FashionMNIST dataset if needed."""
        datasets.FashionMNIST(self.data_dir, train=True, download=True)
        datasets.FashionMNIST(self.data_dir, train=False, download=True)
    
    def setup(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup train, validation, and test dataloaders."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        
        train_dataset = datasets.FashionMNIST(
            self.data_dir, train=True, download=False, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            self.data_dir, train=False, download=False, transform=transform
        )
        
        if self.subset_size:
            train_indices = torch.randperm(len(train_dataset))[:self.subset_size]
            train_dataset = Subset(train_dataset, train_indices)
        
        if self.test_subset_size:
            test_indices = torch.randperm(len(test_dataset))[:self.test_subset_size]
            test_dataset = Subset(test_dataset, test_indices)
        
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create dataloaders (no encoding here - model will handle it)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def get_input_size(self) -> int:
        """Get the raw input size (before encoding)."""
        return self.input_size
    
    def get_num_classes(self) -> int:
        return 10


@register_dataloader("cifar10")
class CIFAR10DataLoader(BaseDataLoader):
    """DataLoader for CIFAR10 dataset."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.data_dir = config.get('data_dir', './data')
        self.batch_size = config.get('batch_size', 128)
        self.subset_size = config.get('subset_size', None)
        self.test_subset_size = config.get('test_subset_size', 1000)
        self.num_workers = config.get('num_workers', 0)
        
        self.input_size = 3072  # CIFAR10 flattened size (3x32x32)
    
    def prepare_data(self):
        """Download CIFAR10 dataset if needed."""
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup train, validation, and test dataloaders."""
        # Flatten RGB images: 3x32x32 = 3072 features
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        
        train_dataset = datasets.CIFAR10(
            self.data_dir, train=True, download=False, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            self.data_dir, train=False, download=False, transform=transform
        )
        
        if self.subset_size:
            train_indices = torch.randperm(len(train_dataset))[:self.subset_size]
            train_dataset = Subset(train_dataset, train_indices)
        
        if self.test_subset_size:
            test_indices = torch.randperm(len(test_dataset))[:self.test_subset_size]
            test_dataset = Subset(test_dataset, test_indices)
        
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create dataloaders (no encoding here - model will handle it)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def get_input_size(self) -> int:
        """Get the raw input size (before encoding)."""
        return self.input_size
    
    def get_num_classes(self) -> int:
        return 10
