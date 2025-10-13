#!/usr/bin/env python3
"""
Trainer for DiffLUT experiments.
Handles model training, validation, and metric logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from typing import Dict, Optional, Callable
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.db_manager import DBManager
from utils.metrics import compute_metrics, list_metrics


class Trainer:
    """
    Trainer class for managing training loops and metric logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
        db_manager: DBManager,
        experiment_id: int,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Training configuration dictionary
            db_manager: Database manager for logging
            experiment_id: ID of the experiment in database
            device: Device to train on (auto-detected if None)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.db_manager = db_manager
        self.experiment_id = experiment_id
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup training parameters
        self.epochs = config.get('epochs', 10)
        self.lr = config.get('lr', 0.001)
        self.optimizer_name = config.get('optimizer', 'adam')
        
        # Setup metrics to track (loss is always tracked)
        self.metric_names = config.get('metrics', ['accuracy'])
        if not isinstance(self.metric_names, list):
            self.metric_names = [self.metric_names]
        
        print(f"Tracking metrics: {self.metric_names}")
        print(f"Available metrics: {list_metrics()}")
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.start_time = None
        self.total_time = 0.0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Gradient tracking settings
        self.save_gradients = config.get('save_gradients', False)
        self.gradient_num_samples = config.get('gradient_num_samples', 100)
        
        # Gradient directory
        if self.save_gradients:
            self.gradient_dir = Path(config.get('gradient_dir', './gradients'))
            self.gradient_dir.mkdir(parents=True, exist_ok=True)
            print(f"Gradient snapshots will be saved to: {self.gradient_dir}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if self.optimizer_name.lower() == 'adam':
            return optim.Adam(trainable_params, lr=self.lr)
        elif self.optimizer_name.lower() == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(trainable_params, lr=self.lr, momentum=momentum)
        elif self.optimizer_name.lower() == 'adamw':
            weight_decay = self.config.get('weight_decay', 0.01)
            return optim.AdamW(trainable_params, lr=self.lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        epoch_start = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track loss and collect outputs for metrics
            total_loss += loss.item()
            all_outputs.append(outputs.detach())
            all_targets.append(targets)
        
        epoch_time = time.time() - epoch_start
        self.total_time += epoch_time
        
        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute configured metrics using registry
        metrics = {'loss': total_loss / len(self.train_loader)}
        computed_metrics = compute_metrics(all_outputs, all_targets, self.metric_names)
        metrics.update(computed_metrics)
        
        # Add timing info
        metrics['epoch_time'] = epoch_time
        metrics['total_time'] = self.total_time
        
        return metrics
    
    def validate(self, epoch: int, loader: DataLoader, phase: str = 'val') -> Dict[str, float]:
        """
        Validate/test the model.
        
        Args:
            epoch: Current epoch number
            loader: Data loader to use
            phase: Phase name ('val' or 'test')
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Collect outputs and targets for metric computation
                all_outputs.append(outputs)
                all_targets.append(targets)
        
        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute configured metrics using registry
        metrics = {'loss': total_loss / len(loader)}
        computed_metrics = compute_metrics(all_outputs, all_targets, self.metric_names)
        metrics.update(computed_metrics)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> str:
        """
        Save best model checkpoint (overwrites previous best for this experiment).
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"exp_{self.experiment_id}_best.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        return str(checkpoint_path)
    
    def compute_and_save_gradients(self, epoch: int) -> Optional[str]:
        """
        Compute and save gradient/weight histograms for validation samples.
        Stores histograms instead of full tensors to save space.
        
        Args:
            epoch: Current epoch
        
        Returns:
            Path to saved gradient file or None if failed
        """
        if not self.save_gradients:
            return None
        
        self.model.eval()
        
        # Collect samples
        samples_collected = 0
        sample_inputs = []
        sample_targets = []
        sample_indices = []
        
        for batch_idx, (inputs, targets) in enumerate(self.val_loader):
            batch_size = inputs.size(0)
            if samples_collected + batch_size > self.gradient_num_samples:
                # Take only what we need
                remaining = self.gradient_num_samples - samples_collected
                inputs = inputs[:remaining]
                targets = targets[:remaining]
                batch_size = remaining
            
            sample_inputs.append(inputs)
            sample_targets.append(targets)
            sample_indices.extend(range(batch_idx * self.val_loader.batch_size, 
                                       batch_idx * self.val_loader.batch_size + batch_size))
            
            samples_collected += batch_size
            
            if samples_collected >= self.gradient_num_samples:
                break
        
        if not sample_inputs:
            return None
        
        # Concatenate samples
        sample_inputs = torch.cat(sample_inputs, dim=0).to(self.device)
        sample_targets = torch.cat(sample_targets, dim=0).to(self.device)
        
        # Enable gradient computation
        sample_inputs.requires_grad = False  # We don't need input gradients
        self.model.train()  # Set to train mode to enable gradients
        
        # Forward pass
        outputs = self.model(sample_inputs)
        loss = self.criterion(outputs, sample_targets)
        
        # Backward pass to compute gradients
        self.model.zero_grad()
        loss.backward()
        
        # Compute histograms for gradients and weights (saves space)
        num_bins = self.config.get('histogram_bins', 256)
        gradient_histograms = {}
        weight_histograms = {}
        
        for name, param in self.model.named_parameters():
            param_np = param.detach().cpu().numpy().ravel()
            
            # Weight histogram
            if param_np.size > 0:
                hist, bin_edges = torch.histogram(
                    torch.from_numpy(param_np), 
                    bins=num_bins
                )
                weight_histograms[name] = {
                    'hist': hist.numpy(),
                    'bin_edges': bin_edges.numpy(),
                    'shape': list(param.shape),
                    'numel': param.numel()
                }
            
            # Gradient histogram
            if param.grad is not None:
                grad_np = param.grad.detach().cpu().numpy().ravel()
                if grad_np.size > 0:
                    hist, bin_edges = torch.histogram(
                        torch.from_numpy(grad_np), 
                        bins=num_bins
                    )
                    gradient_histograms[name] = {
                        'hist': hist.numpy(),
                        'bin_edges': bin_edges.numpy(),
                        'shape': list(param.grad.shape),
                        'numel': param.grad.numel()
                    }
        
        # Save histograms - overwrite for this experiment (one file per experiment)
        gradient_filename = f"exp_{self.experiment_id}_histograms.pkl"
        gradient_path = self.gradient_dir / gradient_filename
        
        # Load existing data if it exists
        import pickle
        if gradient_path.exists():
            with open(gradient_path, 'rb') as f:
                existing_data = pickle.load(f)
        else:
            existing_data = {'epochs': {}}
        
        # Add current epoch data
        existing_data['epochs'][epoch] = {
            'gradient_histograms': gradient_histograms,
            'weight_histograms': weight_histograms,
            'num_samples': len(sample_indices),
            'loss': loss.item()
        }
        existing_data['experiment_id'] = self.experiment_id
        existing_data['num_bins'] = num_bins
        
        # Save updated data
        with open(gradient_path, 'wb') as f:
            pickle.dump(existing_data, f)
        
        # Reset model to eval mode
        self.model.eval()
        
        return str(gradient_path)
    
    def train(self):
        """
        Main training loop.
        """
        print("="*60)
        print("Starting Training")
        print("="*60)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Epochs: {self.epochs}")
        print(f"Optimizer: {self.optimizer_name}")
        print(f"Learning Rate: {self.lr}")
        print(f"Device: {self.device}")
        print(f"Save Histograms: {self.save_gradients}")
        if self.save_gradients:
            print(f"Histogram Samples: {self.gradient_num_samples}")
            print(f"Histogram Bins: {self.config.get('histogram_bins', 256)}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        if trainable_params == 0:
            print("ERROR: No trainable parameters!")
            self.db_manager.update_experiment_status(self.experiment_id, 'failed')
            return
        
        print("="*60)
        
        # Update experiment status
        self.db_manager.update_experiment_status(
            self.experiment_id, 'running', str(self.device)
        )
        
        self.start_time = time.time()
        
        try:
            for epoch in range(1, self.epochs + 1):
                # Train
                train_metrics = self.train_epoch(epoch)
                
                # Log training metrics
                train_mp = self.db_manager.create_measurement_point(
                    experiment_id=self.experiment_id,
                    epoch=epoch,
                    phase='train',
                    epoch_time=train_metrics['epoch_time'],
                    total_time=train_metrics['total_time']
                )
                
                # Extract only the metrics to log (exclude timing info)
                train_metrics_to_log = {k: v for k, v in train_metrics.items() 
                                       if k not in ['epoch_time', 'total_time']}
                self.db_manager.add_metrics(train_mp.id, train_metrics_to_log)
                
                # Validate
                val_metrics = self.validate(epoch, self.val_loader, 'val')
                
                # Check if best model (use primary metric, default to accuracy)
                primary_metric = self.config.get('primary_metric', 'accuracy')
                if primary_metric not in val_metrics:
                    primary_metric = 'accuracy'  # Fallback
                
                is_best = False
                if primary_metric in val_metrics:
                    current_value = val_metrics[primary_metric]
                    is_best = current_value > self.best_val_acc
                    if is_best:
                        self.best_val_acc = current_value
                        self.best_epoch = epoch
                
                # Save checkpoint if best
                checkpoint_path = None
                if is_best:
                    checkpoint_path = self.save_checkpoint(epoch, val_metrics)
                
                # Log validation metrics
                val_mp = self.db_manager.create_measurement_point(
                    experiment_id=self.experiment_id,
                    epoch=epoch,
                    phase='val'
                )
                self.db_manager.add_metrics(
                    val_mp.id, 
                    val_metrics,
                    checkpoint_path=checkpoint_path,
                    is_best=is_best
                )
                
                # Compute and save gradients if enabled
                if self.save_gradients:
                    gradient_path = self.compute_and_save_gradients(epoch)
                    if gradient_path:
                        self.db_manager.add_gradient_snapshot(
                            measurement_point_id=val_mp.id,
                            gradient_path=gradient_path,
                            num_samples=self.gradient_num_samples,
                            sample_indices=None,  # Could store indices if needed
                            metadata={'loss': val_metrics.get('loss')}
                        )
                
                # Print progress
                train_str = f"Loss: {train_metrics['loss']:.4f}"
                for metric_name in self.metric_names:
                    if metric_name in train_metrics:
                        train_str += f" {metric_name}: {train_metrics[metric_name]:.2f}"
                
                val_str = f"Loss: {val_metrics['loss']:.4f}"
                for metric_name in self.metric_names:
                    if metric_name in val_metrics:
                        val_str += f" {metric_name}: {val_metrics[metric_name]:.2f}"
                
                print(f"Epoch {epoch}/{self.epochs} ({train_metrics['epoch_time']:.1f}s): "
                      f"Train {train_str} | Val {val_str}"
                      f"{' [BEST]' if is_best else ''}")
            
            # Final test evaluation
            print("\n" + "="*60)
            print("Running Final Test Evaluation")
            print("="*60)
            test_metrics = self.validate(self.epochs, self.test_loader, 'test')
            
            # Log test metrics
            test_mp = self.db_manager.create_measurement_point(
                experiment_id=self.experiment_id,
                epoch=self.epochs,
                phase='test'
            )
            self.db_manager.add_metrics(test_mp.id, test_metrics)
            
            # Print test results
            test_str = f"Test Loss: {test_metrics['loss']:.4f}"
            for metric_name, value in test_metrics.items():
                if metric_name != 'loss':
                    test_str += f" {metric_name}: {value:.2f}"
            print(test_str)
            
            primary_metric = self.config.get('primary_metric', 'accuracy')
            print(f"Best Val {primary_metric}: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
            print(f"Total Time: {self.total_time:.1f}s")
            print("="*60)
            
            # Update experiment status
            self.db_manager.update_experiment_status(self.experiment_id, 'completed')
            
        except Exception as e:
            print(f"ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            self.db_manager.update_experiment_status(self.experiment_id, 'failed')
            raise
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
