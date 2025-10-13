#!/usr/bin/env python3
"""
Base dataloader class and registry for experiment dataloaders.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict
from torch.utils.data import DataLoader

# Registry for dataloaders
_DATALOADER_REGISTRY = {}


def register_dataloader(name: str):
    """
    Decorator to register a dataloader class.
    
    Args:
        name: Name to register the dataloader under
    """
    def decorator(cls):
        _DATALOADER_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataloader(name: str, config: dict):
    """
    Get a dataloader instance by name.
    
    Args:
        name: Name of the registered dataloader
        config: Configuration dictionary for the dataloader
    
    Returns:
        Dataloader instance
    """
    if name not in _DATALOADER_REGISTRY:
        raise ValueError(f"Dataloader '{name}' not found. Available: {list(_DATALOADER_REGISTRY.keys())}")
    
    dataloader_class = _DATALOADER_REGISTRY[name]
    return dataloader_class(config)


class BaseDataLoader(ABC):
    """
    Abstract base class for all dataloaders.
    Provides interface for data preparation, setup, and access.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the dataloader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    @abstractmethod
    def prepare_data(self):
        """
        Download and prepare data if needed.
        This is called only once, typically for downloading.
        """
        pass
    
    @abstractmethod
    def setup(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Setup and return dataloaders for train, validation, and test.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        pass
    
    @abstractmethod
    def get_input_size(self) -> int:
        """
        Get the input feature size after encoding.
        
        Returns:
            Input feature dimension
        """
        pass
    
    @abstractmethod
    def get_num_classes(self) -> int:
        """
        Get the number of output classes.
        
        Returns:
            Number of classes
        """
        pass
