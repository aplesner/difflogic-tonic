#!/usr/bin/env python3
"""
Common metrics for experiment evaluation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List
import numpy as np

from . import register_metric


@register_metric("accuracy")
def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy.
    
    Args:
        outputs: Model outputs (N, num_classes)
        targets: Ground truth labels (N,)
    
    Returns:
        Accuracy as percentage
    """
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


@register_metric("top_k_accuracy")
def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        outputs: Model outputs (N, num_classes)
        targets: Ground truth labels (N,)
        k: Top k predictions to consider
    
    Returns:
        Top-k accuracy as percentage
    """
    _, topk_preds = outputs.topk(k, dim=1, largest=True, sorted=True)
    targets_expanded = targets.view(-1, 1).expand_as(topk_preds)
    correct = topk_preds.eq(targets_expanded).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


@register_metric("perplexity")
def perplexity(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute perplexity.
    
    Args:
        outputs: Model outputs (N, num_classes)
        targets: Ground truth labels (N,)
    
    Returns:
        Perplexity value
    """
    log_probs = F.log_softmax(outputs, dim=1)
    nll = F.nll_loss(log_probs, targets, reduction='mean')
    return torch.exp(nll).item()


@register_metric("precision_recall_f1")
def precision_recall_f1(outputs: torch.Tensor, targets: torch.Tensor, 
                        num_classes: int = 10, average: str = 'macro') -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        outputs: Model outputs (N, num_classes)
        targets: Ground truth labels (N,)
        num_classes: Number of classes
        average: Averaging method ('macro', 'micro', 'weighted')
    
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    _, predicted = outputs.max(1)
    
    # Convert to numpy for easier manipulation
    pred_np = predicted.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    if average == 'micro':
        # Micro average: calculate metrics globally
        tp = (pred_np == target_np).sum()
        fp = (pred_np != target_np).sum()
        fn = fp  # In multi-class, FP = FN for micro averaging
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
    else:
        # Macro or weighted average: calculate per class
        precisions = []
        recalls = []
        f1s = []
        weights = []
        
        for c in range(num_classes):
            tp = ((pred_np == c) & (target_np == c)).sum()
            fp = ((pred_np == c) & (target_np != c)).sum()
            fn = ((pred_np != c) & (target_np == c)).sum()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1_c)
            weights.append((target_np == c).sum())
        
        if average == 'macro':
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)
        else:  # weighted
            total_weight = sum(weights)
            precision = sum(p * w for p, w in zip(precisions, weights)) / total_weight
            recall = sum(r * w for r, w in zip(recalls, weights)) / total_weight
            f1 = sum(f * w for f, w in zip(f1s, weights)) / total_weight
    
    return {
        'precision': precision * 100.0,
        'recall': recall * 100.0,
        'f1': f1 * 100.0
    }


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor, 
                   metric_names: List[str], num_classes: int = 10) -> Dict[str, float]:
    """
    Compute multiple metrics.
    
    Args:
        outputs: Model outputs (N, num_classes)
        targets: Ground truth labels (N,)
        metric_names: List of metric names to compute
        num_classes: Number of classes
    
    Returns:
        Dictionary mapping metric names to values
    """
    results = {}
    
    for metric_name in metric_names:
        if metric_name == 'accuracy':
            results['accuracy'] = accuracy(outputs, targets)
        elif metric_name == 'top_5_accuracy':
            results['top_5_accuracy'] = top_k_accuracy(outputs, targets, k=5)
        elif metric_name == 'top_10_accuracy':
            results['top_10_accuracy'] = top_k_accuracy(outputs, targets, k=min(10, num_classes))
        elif metric_name == 'perplexity':
            results['perplexity'] = perplexity(outputs, targets)
        elif metric_name in ['precision', 'recall', 'f1']:
            prf = precision_recall_f1(outputs, targets, num_classes)
            if metric_name == 'precision':
                results['precision'] = prf['precision']
            elif metric_name == 'recall':
                results['recall'] = prf['recall']
            elif metric_name == 'f1':
                results['f1'] = prf['f1']
    
    return results


class MetricsTracker:
    """Track metrics across epochs."""
    
    def __init__(self, metric_names: List[str]):
        """
        Initialize tracker.
        
        Args:
            metric_names: List of metrics to track
        """
        self.metric_names = metric_names
        self.history = {name: [] for name in metric_names}
        self.history['loss'] = []
    
    def update(self, metrics: Dict[str, float], loss: float = None):
        """
        Update tracker with new metrics.
        
        Args:
            metrics: Dictionary of metric values
            loss: Loss value (optional)
        """
        for name in self.metric_names:
            if name in metrics:
                self.history[name].append(metrics[name])
        
        if loss is not None:
            self.history['loss'].append(loss)
    
    def get_best(self, metric_name: str = 'accuracy', mode: str = 'max') -> tuple:
        """
        Get best value and epoch for a metric.
        
        Args:
            metric_name: Name of metric
            mode: 'max' or 'min'
        
        Returns:
            Tuple of (best_value, best_epoch)
        """
        if metric_name not in self.history or not self.history[metric_name]:
            return None, None
        
        values = self.history[metric_name]
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return values[best_idx], best_idx
    
    def get_latest(self, metric_name: str = 'accuracy') -> float:
        """
        Get latest value for a metric.
        
        Args:
            metric_name: Name of metric
        
        Returns:
            Latest value or None
        """
        if metric_name not in self.history or not self.history[metric_name]:
            return None
        return self.history[metric_name][-1]
    
    def summary(self) -> Dict[str, any]:
        """
        Get summary of all metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        for name, values in self.history.items():
            if not values:
                continue
            summary[name] = {
                'latest': values[-1],
                'best': max(values) if name != 'loss' and name != 'perplexity' else min(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        return summary
