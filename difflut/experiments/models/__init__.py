"""
Model definitions for experiments.
"""

from typing import Dict, Type
import torch.nn as nn

# Model registry
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """
    Decorator to register a model class.
    
    Args:
        name: Name to register the model under
        
    Example:
        @register_model("layered_feedforward")
        class LayeredFeedForward(nn.Module):
            pass
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str) -> Type[nn.Module]:
    """
    Get a registered model class by name.
    
    Args:
        name: Name of the model
        
    Returns:
        Model class
        
    Raises:
        ValueError: If model not found
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found. "
            f"Available models: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]


def list_models() -> list:
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys())


def build_model(config: dict, input_size: int, num_classes: int) -> nn.Module:
    """
    Build a model from configuration.
    
    Args:
        config: Model configuration dictionary
        input_size: Size of input features
        num_classes: Number of output classes
    
    Returns:
        Instantiated model
    """
    model_name = config.get('name', 'LayeredFeedForward')
    model_params = config.get('params', {})
    
    model_class = get_model(model_name)
    return model_class(model_params, input_size, num_classes)


# Import models (this will trigger registration)
from models.layered_feedforward import LayeredFeedForward

__all__ = [
    'LayeredFeedForward',
    'register_model',
    'get_model',
    'list_models',
    'build_model'
]
