import time
import logging

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from . import config
from . import io_funcs

logger = logging.getLogger(__name__)

def get_correct_and_accuracy(output: torch.Tensor, target: torch.Tensor) -> tuple[int, float]:
    """Compute accuracy"""
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = int((preds == target).sum().item())
        return correct, correct / target.size(0)


def train_step(
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        config: config.Config,
        batch_idx: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[float, float]:
    """Perform a single training step"""
    data, targets = batch

    data, targets = data.to(device, dtype=dtype), targets.to(device)

    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # get sample accuracy
    _, accuracy = get_correct_and_accuracy(outputs, targets)

    # if config.base.debug and batch_idx % config.train.log_interval == 0:
    #     logger.debug(f"Device, dtype and shape of outputs: {outputs.device}, {outputs.dtype}, {outputs.shape}")

    return loss.item(), accuracy


def train_epoch(
        model: nn.Module, 
        dataloader: DataLoader, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        batch_count: int, 
        config: config.Config, 
        start_time: float, 
        last_checkpoint_time: float
    ) -> tuple[float, int, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    current_time = time.time()
    checkpoint_interval_seconds = config.train.checkpoint_interval_minutes * 60
    device = config.train.device if isinstance(config.train.device, torch.device) else torch.device(config.train.device)
    dtype = config.train.dtype if isinstance(config.train.dtype, torch.dtype) else getattr(torch, config.train.dtype)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', dtype=dtype):
        for batch_idx, batch in enumerate(dataloader):
            loss, accuracy = train_step(
                model=model, batch=batch, criterion=criterion, optimizer=optimizer, config=config, batch_idx=batch_idx, device=device, dtype=dtype
            )
            total_loss += loss
            batch_count += 1

            # Time-based checkpointing
            current_time = time.time()
            if current_time - last_checkpoint_time >= checkpoint_interval_seconds:
                elapsed_time = current_time - start_time
                logger.info(f"Batch {batch_count}, Loss: {loss:.4f}, Elapsed: {elapsed_time/60:.1f} minutes - Saving checkpoint...")
                io_funcs.save_checkpoint(model=model, optimizer=optimizer, batch_count=batch_count, config=config, job_id=config.base.job_id, elapsed_time=elapsed_time)
                last_checkpoint_time = current_time

            if batch_idx % config.train.log_interval == 0:
                logger.info(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss:.4f}, Accuracy: {100 * accuracy:.2f}%")

            if config.base.debug and batch_idx >= config.train.debugging_steps:
                break  # For debugging, limit to specified batches per epoch

    return total_loss / len(dataloader), batch_count, last_checkpoint_time


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, config: config.Config) -> tuple[float, float]:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    corrects, total = 0, 0
    dtype = config.train.dtype if isinstance(config.train.dtype, torch.dtype) else getattr(torch, config.train.dtype)
    device = config.train.device if isinstance(config.train.device, torch.device) else torch.device(config.train.device)

    with torch.no_grad() and torch.autocast(device.type if device.type != 'mps' else 'cpu'):
        for data_batch, targets in dataloader:
            data_batch, targets = data_batch.to(device, dtype=dtype), targets.to(device)
            outputs = model(data_batch)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            correct, _ = get_correct_and_accuracy(outputs, targets)
            corrects += correct
            total += targets.size(0)

    accuracy = 100. * corrects / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy