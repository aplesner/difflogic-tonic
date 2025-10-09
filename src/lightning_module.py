"""PyTorch Lightning module for training models

This module wraps existing models (MLP, CNN, DiffLogic) in a LightningModule
for simplified training, evaluation, and logging.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torchmetrics import Accuracy

from . import config

logger = logging.getLogger(__name__)


class LitModel(L.LightningModule):
    """Lightning wrapper for models

    Handles:
    - Training and validation steps
    - Automatic mixed precision
    - Metric logging (loss, accuracy)
    - Optimizer configuration
    - Learning rate scheduling (optional)
    """

    def __init__(self, model: nn.Module, cfg: config.Config, num_classes: int):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()

        # Metrics - using torchmetrics for proper distributed computation
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dtype conversion

        Args:
            x: Input tensor, may be bool or float

        Returns:
            Model output logits
        """
        # Convert boolean data to float if necessary
        if x.dtype == torch.bool:
            x = x.float()
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step

        Args:
            batch: Tuple of (data, targets)
            batch_idx: Batch index

        Returns:
            Loss tensor for backpropagation
        """
        data, targets = batch
        outputs = self(data)
        loss = self.criterion(outputs, targets)

        # Compute accuracy - use .detach() not .item() per Lightning best practices
        preds = outputs.argmax(dim=1)
        acc = self.train_acc(preds, targets)

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step

        Args:
            batch: Tuple of (data, targets)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        data, targets = batch
        outputs = self(data)
        loss = self.criterion(outputs, targets)

        # Compute accuracy
        preds = outputs.argmax(dim=1)
        acc = self.val_acc(preds, targets)

        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step (same as validation)

        Args:
            batch: Tuple of (data, targets)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        data, targets = batch
        outputs = self(data)
        loss = self.criterion(outputs, targets)

        # Compute accuracy
        preds = outputs.argmax(dim=1)
        acc = self.val_acc(preds, targets)  # Reuse val_acc for test

        # Log with test/ prefix
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test/acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and optional learning rate scheduler

        Returns:
            Optimizer or dict with optimizer and scheduler
        """
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.cfg.train.learning_rate
        )

        # Add scheduler if configured
        scheduler_cfg = self.cfg.train.scheduler
        if scheduler_cfg.enabled:
            if scheduler_cfg.type == "step":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_cfg.step_size,
                    gamma=scheduler_cfg.gamma
                )
            elif scheduler_cfg.type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.cfg.train.epochs,
                    eta_min=scheduler_cfg.min_lr
                )
            elif scheduler_cfg.type == "reduce_on_plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',  # maximize validation accuracy
                    factor=scheduler_cfg.factor,
                    patience=scheduler_cfg.patience,
                    min_lr=scheduler_cfg.min_lr
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val/acc',
                        'interval': 'epoch',
                        'frequency': 1
                    }
                }
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_cfg.type}")

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

        return optimizer
