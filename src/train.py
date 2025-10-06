import time
import logging

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from . import config
from . import io_funcs

logger = logging.getLogger(__name__)

def train_step(
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device
    ) -> tuple[float, float]:
    """Perform a single training step"""
    data, targets = batch

    data, targets = data.to(device), targets.to(device)

    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # get sample accuracy
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    accuracy = correct / targets.size(0)

    return loss.item(), accuracy


def train_epoch(
        model: nn.Module, 
        dataloader: DataLoader, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        device: torch.device, 
        batch_count: int, 
        job_id: str,
        config: config.Config, 
        start_time: float, 
        last_checkpoint_time: float
    ) -> tuple[float, int, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    current_time = time.time()
    checkpoint_interval_seconds = config.train.checkpoint_interval_minutes * 60

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', dtype=torch.bfloat16):
        for batch_idx, (data_batch, targets) in enumerate(dataloader):
            loss, accuracy = train_step(model, (data_batch, targets), criterion, optimizer, device)
            total_loss += loss
            batch_count += 1

            # Time-based checkpointing
            current_time = time.time()
            if current_time - last_checkpoint_time >= checkpoint_interval_seconds:
                elapsed_time = current_time - start_time
                logger.info(f"Batch {batch_count}, Loss: {loss:.4f}, Elapsed: {elapsed_time/60:.1f} minutes - Saving checkpoint...")
                io_funcs.save_checkpoint(model=model, optimizer=optimizer, batch_count=batch_count, config=config, job_id=job_id, elapsed_time=elapsed_time)
                last_checkpoint_time = current_time

            if batch_idx % config.train.log_interval == 0:
                logger.info(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            if batch_idx >= 10 and config.base.debug:
                break  # For debugging, limit to 10 batches per epoch

    return total_loss / len(dataloader), batch_count, last_checkpoint_time


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad() and torch.autocast(device.type if device.type != 'mps' else 'cpu'):
        for data_batch, targets in dataloader:
            data_batch, targets = data_batch.to(device, dtype=torch.bfloat16), targets.to(device)
            outputs = model(data_batch)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy