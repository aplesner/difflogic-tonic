#!/usr/bin/env python3
"""
Diagnostic script to identify why MLP/CNN aren't converging.
This simulates the exact training pipeline but with detailed logging.

Usage:
    python3 diagnose_training.py configs/nmnist_mlp.yaml
    python3 diagnose_training.py configs/nmnist_cnn.yaml
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Import from the actual codebase
from src.config import Config
from src.data import get_dataloaders
from src.model import create_model
from src.helpers import setup_seed, get_model_input_shape, get_num_classes


def diagnose_batch(model, batch, criterion, device, dtype):
    """Diagnose a single batch"""
    data, targets = batch

    print(f"\n  Raw data: dtype={data.dtype}, shape={data.shape}")
    print(f"  Raw data: min={data.float().min():.3f}, max={data.float().max():.3f}, mean={data.float().mean():.3f}")

    # Simulate train_step conversion
    if data.dtype == torch.bool:
        data = data.float()
        print(f"  After .float(): dtype={data.dtype}")

    data, targets = data.to(device, dtype=dtype), targets.to(device)
    print(f"  After .to(device, dtype): dtype={data.dtype}, device={data.device}")

    # Forward pass
    with torch.no_grad(), torch.autocast(device.type if device.type != 'mps' else 'cpu', dtype=dtype):
        outputs = model(data)
        loss = criterion(outputs, targets)

    pred = outputs.argmax(dim=1)
    acc = (pred == targets).float().mean().item()

    print(f"  Output: shape={outputs.shape}, dtype={outputs.dtype}")
    print(f"  Output stats: min={outputs.min():.3f}, max={outputs.max():.3f}, mean={outputs.mean():.3f}")
    print(f"  Loss: {loss.item():.4f}, Accuracy: {acc*100:.2f}%")

    # Check for abnormalities
    if torch.isnan(outputs).any():
        print(f"  ⚠ WARNING: NaN values in output!")
    if torch.isinf(outputs).any():
        print(f"  ⚠ WARNING: Inf values in output!")

    return loss.item(), acc


def train_diagnostic(config_file, num_epochs=3, max_batches=50):
    """Run diagnostic training"""
    print("="*80)
    print(f"DIAGNOSTIC TRAINING: {config_file}")
    print("="*80)

    # Load config
    cfg = Config.from_yaml(config_file)
    setup_seed(cfg.base.seed)

    print(f"\nConfiguration:")
    print(f"  Model type: {cfg.model.model_type}")
    print(f"  Learning rate: {cfg.train.learning_rate}")
    print(f"  Batch size: {cfg.train.dataloader.batch_size}")
    print(f"  Device: {cfg.train.device}")
    print(f"  Dtype: {cfg.train.dtype}")
    print(f"  Debugging steps: {cfg.train.debugging_steps}")
    print(f"  Debug mode: {cfg.base.debug}")

    # Get dataloaders
    print(f"\nLoading data...")
    train_dataloader, test_dataloader = get_dataloaders(cfg)
    print(f"  Train batches: {len(train_dataloader)}")
    print(f"  Test batches: {len(test_dataloader)}")

    # Create model
    input_shape = get_model_input_shape(cfg.data)
    num_classes = get_num_classes(cfg.data)
    print(f"\nCreating model...")
    print(f"  Input shape: {input_shape}")
    print(f"  Num classes: {num_classes}")

    model = create_model(config=cfg, input_shape=input_shape, num_classes=num_classes)
    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    device = cfg.train.device if isinstance(cfg.train.device, torch.device) else torch.device(cfg.train.device)
    dtype = cfg.train.dtype if isinstance(cfg.train.dtype, torch.dtype) else getattr(torch, cfg.train.dtype)

    print(f"\nDiagnosing first batch...")
    first_batch = next(iter(train_dataloader))
    diagnose_batch(model, first_batch, criterion, device, dtype)

    # Training loop
    print(f"\n{'='*80}")
    print(f"Training for {num_epochs} epochs (max {max_batches} batches per epoch)")
    print(f"{'='*80}")

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_accs = []

        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= max_batches:
                break

            data, targets = batch

            # Convert bool to float
            if data.dtype == torch.bool:
                data = data.float()

            data, targets = data.to(device, dtype=dtype), targets.to(device)

            optimizer.zero_grad()

            # Use autocast like in train.py
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', dtype=dtype):
                outputs = model(data)
                loss = criterion(outputs, targets)

            loss.backward()

            # Check gradients
            max_grad = 0
            for p in model.parameters():
                if p.grad is not None:
                    max_grad = max(max_grad, p.grad.abs().max().item())

            optimizer.step()

            # Calculate accuracy
            pred = outputs.argmax(dim=1)
            acc = (pred == targets).float().mean().item()

            epoch_losses.append(loss.item())
            epoch_accs.append(acc)

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{max_batches}: "
                      f"Loss={loss.item():.4f}, Acc={acc*100:.2f}%, MaxGrad={max_grad:.2e}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_acc = sum(epoch_accs) / len(epoch_accs) * 100

        # Evaluate on test set (small sample)
        model.eval()
        test_losses = []
        test_accs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                if batch_idx >= 10:  # Only test on 10 batches
                    break

                data, targets = batch

                if data.dtype == torch.bool:
                    data = data.float()

                data, targets = data.to(device, dtype=dtype), targets.to(device)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
                    outputs = model(data)
                    loss = criterion(outputs, targets)

                pred = outputs.argmax(dim=1)
                acc = (pred == targets).float().mean().item()

                test_losses.append(loss.item())
                test_accs.append(acc)

        test_loss = sum(test_losses) / len(test_losses)
        test_acc = sum(test_accs) / len(test_accs) * 100

        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    Train: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")
        print(f"    Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")

        # Check for convergence issues
        if avg_loss > 10:
            print(f"    ⚠ WARNING: Loss is very high (>10)")
        if test_acc < 15:
            print(f"    ⚠ WARNING: Test accuracy is near random guessing")
        elif test_acc > 30:
            print(f"    ✓ Model is learning (>30% accuracy)")

    print(f"\n{'='*80}")
    print(f"Diagnosis Complete")
    print(f"{'='*80}")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    if test_acc > 30:
        print("✓ SUCCESS: Model is learning properly")
    else:
        print("✗ PROBLEM: Model is not learning well")
        print("\nPossible issues:")
        print("  1. Learning rate too low/high")
        print("  2. Model architecture issue")
        print("  3. Data preprocessing problem")
        print("  4. Gradient flow issue")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 diagnose_training.py <config_file>")
        print("\nExamples:")
        print("  python3 diagnose_training.py configs/nmnist_mlp.yaml")
        print("  python3 diagnose_training.py configs/nmnist_cnn.yaml")
        sys.exit(1)

    config_file = sys.argv[1]
    if not Path(config_file).exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)

    train_diagnostic(config_file)
