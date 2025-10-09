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
        self.num_classes = num_classes

        # Metrics - using torchmetrics for proper distributed computation
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # For periodic detailed logging
        self.log_output_every_n_steps = cfg.train.lightning.log_every_n_steps * 5  # Log outputs less frequently

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

        # Periodic detailed logging of outputs
        if batch_idx % self.log_output_every_n_steps == 0:
            self._log_output_statistics(outputs, targets, prefix='train')

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

        # Log detailed statistics for first batch of each validation epoch
        if batch_idx == 0:
            self._log_output_statistics(outputs, targets, prefix='val')

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

    def _log_output_statistics(self, outputs: torch.Tensor, targets: torch.Tensor, prefix: str):
        """Log detailed statistics about model outputs

        Args:
            outputs: Model output logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            prefix: Logging prefix (e.g., 'train', 'val')
        """
        with torch.no_grad():
            # Compute softmax probabilities
            probs = torch.softmax(outputs, dim=1)

            # Logit statistics
            logit_mean = outputs.mean()
            logit_std = outputs.std()
            logit_min = outputs.min()
            logit_max = outputs.max()

            # Probability statistics
            prob_mean = probs.mean()
            prob_std = probs.std()

            # Confidence statistics (max probability per sample)
            max_probs, _ = probs.max(dim=1)
            confidence_mean = max_probs.mean()
            confidence_std = max_probs.std()
            confidence_min = max_probs.min()
            confidence_max = max_probs.max()

            # Per-class prediction distribution
            preds = outputs.argmax(dim=1)

            # Ground truth class distribution
            target_counts = torch.bincount(targets, minlength=self.num_classes).float()
            target_distribution = target_counts / targets.size(0)

            # Prediction class distribution
            pred_counts = torch.bincount(preds, minlength=self.num_classes).float()
            pred_distribution = pred_counts / preds.size(0)

            # Entropy of predictions (measure of uncertainty)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()

            # Log scalar statistics
            self.log(f'{prefix}/logit_mean', logit_mean, on_step=False, on_epoch=True, logger=True)
            self.log(f'{prefix}/logit_std', logit_std, on_step=False, on_epoch=True, logger=True)
            self.log(f'{prefix}/logit_min', logit_min, on_step=False, on_epoch=True, logger=True)
            self.log(f'{prefix}/logit_max', logit_max, on_step=False, on_epoch=True, logger=True)

            self.log(f'{prefix}/prob_mean', prob_mean, on_step=False, on_epoch=True, logger=True)
            self.log(f'{prefix}/prob_std', prob_std, on_step=False, on_epoch=True, logger=True)

            self.log(f'{prefix}/confidence_mean', confidence_mean, on_step=False, on_epoch=True, logger=True)
            self.log(f'{prefix}/confidence_std', confidence_std, on_step=False, on_epoch=True, logger=True)
            self.log(f'{prefix}/confidence_min', confidence_min, on_step=False, on_epoch=True, logger=True)
            self.log(f'{prefix}/confidence_max', confidence_max, on_step=False, on_epoch=True, logger=True)

            self.log(f'{prefix}/entropy', entropy, on_step=False, on_epoch=True, logger=True)

            # Log to console periodically
            if prefix == 'train':
                logger.debug(
                    f"Output stats - Logits: [{logit_min:.3f}, {logit_max:.3f}] "
                    f"(μ={logit_mean:.3f}, σ={logit_std:.3f}), "
                    f"Confidence: μ={confidence_mean:.3f}, σ={confidence_std:.3f}, "
                    f"Entropy: {entropy:.3f}"
                )

            # Log distributions to WandB if available
            if self.logger and hasattr(self.logger, 'experiment'):
                try:
                    # Log histograms for WandB
                    self.logger.experiment.log({
                        f'{prefix}/logits_histogram': outputs.detach().cpu().numpy().flatten(),
                        f'{prefix}/probs_histogram': probs.detach().cpu().numpy().flatten(),
                        f'{prefix}/confidence_histogram': max_probs.detach().cpu().numpy(),
                        f'{prefix}/target_distribution': target_distribution.detach().cpu().numpy(),
                        f'{prefix}/pred_distribution': pred_distribution.detach().cpu().numpy(),
                    })
                except Exception as e:
                    # Silently fail if logging to WandB fails (e.g., offline mode)
                    pass

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
