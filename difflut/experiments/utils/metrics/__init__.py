"""
Metrics for experiments.
"""

from typing import Dict, Callable
import torch

# Metric registry
_METRIC_REGISTRY: Dict[str, Callable] = {}


def register_metric(name: str):
    """
    Decorator to register a metric function.
    
    Args:
        name: Name to register the metric under
        
    Example:
        @register_metric("accuracy")
        def accuracy(outputs, targets):
            pass
    """
    def decorator(func: Callable) -> Callable:
        if name in _METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered")
        _METRIC_REGISTRY[name] = func
        return func
    return decorator


def get_metric(name: str) -> Callable:
    """
    Get a registered metric function by name.
    
    Args:
        name: Name of the metric
        
    Returns:
        Metric function
        
    Raises:
        ValueError: If metric not found
    """
    if name not in _METRIC_REGISTRY:
        raise ValueError(
            f"Metric '{name}' not found. "
            f"Available metrics: {list(_METRIC_REGISTRY.keys())}"
        )
    return _METRIC_REGISTRY[name]


def list_metrics() -> list:
    """
    List all registered metrics.
    
    Returns:
        List of metric names
    """
    return list(_METRIC_REGISTRY.keys())


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor, 
                   metric_names: list) -> Dict[str, float]:
    """
    Compute multiple metrics using the registry.
    
    Args:
        outputs: Model outputs (N, num_classes)
        targets: Ground truth labels (N,)
        metric_names: List of metric names to compute
    
    Returns:
        Dictionary mapping metric names to values
    """
    results = {}
    
    for metric_name in metric_names:
        if metric_name in _METRIC_REGISTRY:
            metric_fn = _METRIC_REGISTRY[metric_name]
            value = metric_fn(outputs, targets)
            
            # Handle metrics that return dictionaries
            if isinstance(value, dict):
                results.update(value)
            else:
                results[metric_name] = value
        else:
            print(f"Warning: Metric '{metric_name}' not found in registry. Skipping.")
    
    return results


# Import classification metrics to register them
from . import classification_metrics

__all__ = [
    'register_metric',
    'get_metric',
    'list_metrics',
    'compute_metrics',
]


def list_metrics() -> list:
    """List all registered metric names."""
    return list(_METRIC_REGISTRY.keys())


# Import metrics (this will trigger registration)
from .classification_metrics import (
    accuracy,
    top_k_accuracy,
    perplexity,
    precision_recall_f1,
    compute_metrics,
    MetricsTracker
)

__all__ = [
    'accuracy',
    'top_k_accuracy',
    'perplexity',
    'precision_recall_f1',
    'compute_metrics',
    'MetricsTracker',
    'register_metric',
    'get_metric',
    'list_metrics'
]
