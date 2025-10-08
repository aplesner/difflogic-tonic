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

    # Convert boolean data to float if necessary
    if data.dtype == torch.bool:
        data = data.float()

    data, targets = data.to(device), targets.to(device)
    device_type = device.type if device.type != 'mps' else 'cpu'

    with torch.autocast(device_type=device_type, dtype=dtype):
        outputs = model(data)
        loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # get sample accuracy
    correct, accuracy = get_correct_and_accuracy(outputs, targets)

    if config.base.debug:# and batch_idx % config.train.log_interval == 0:
        # print data, target and output stats for debugging
        # logger.debug(f"Batch {batch_idx}:")
        # print(targets.cpu().numpy().tolist())
        # # print the output tensor as indices of the max logits
        # print(outputs.argmax(dim=1).cpu().numpy().tolist())
        
        logger.debug(f"  Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.2f}%")
        # Uncomment the following line to log output tensor details

    return loss.item(), correct, accuracy


def train_epoch(
        model: nn.Module, 
        dataloader: DataLoader, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        batch_count: int, 
        config: config.Config, 
        start_time: float, 
        last_checkpoint_time: float
    ) -> tuple[float, float, int, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    current_time = time.time()
    checkpoint_interval_seconds = config.train.checkpoint_interval_minutes * 60
    device = config.train.device if isinstance(config.train.device, torch.device) else torch.device(config.train.device)
    dtype = config.train.dtype if isinstance(config.train.dtype, torch.dtype) else getattr(torch, config.train.dtype)

    correct, total_samples = 0, 0

    for batch_idx, batch in enumerate(dataloader):
        loss, batch_correct, accuracy = train_step(
            model=model, batch=batch, criterion=criterion, optimizer=optimizer, config=config, batch_idx=batch_idx, device=device, dtype=dtype
        )
        batch_size = batch[0].shape[0]
        total_loss += loss
        correct += batch_correct
        total_samples += batch_size

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
    batch_count += total_samples
    accuracy = 100. * correct / total_samples if total_samples > 0 else 0.
    loss_per_sample = total_loss / total_samples if total_samples > 0 else 0.
    return loss_per_sample, accuracy, batch_count, last_checkpoint_time


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, config: config.Config) -> tuple[float, float]:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    corrects, total_samples = 0, 0
    dtype = config.train.dtype if isinstance(config.train.dtype, torch.dtype) else getattr(torch, config.train.dtype)
    device = config.train.device if isinstance(config.train.device, torch.device) else torch.device(config.train.device)

    device_type = device.type if device.type != 'mps' else 'cpu'

    for data_batch, targets in dataloader:
        # Convert boolean data to float if necessary
        if data_batch.dtype == torch.bool:
            data_batch = data_batch.float()

        data_batch, targets = data_batch.to(device), targets.to(device)
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=dtype):
            outputs = model(data_batch)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        correct, _ = get_correct_and_accuracy(outputs, targets)
        corrects += correct
        total_samples += targets.size(0)

    accuracy = 100. * corrects / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    return avg_loss, accuracy