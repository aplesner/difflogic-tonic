#!/usr/bin/env python3
"""
Minimal Working Example for NMNIST Training

This standalone script validates the dataset and training pipeline without
dependencies on the main codebase. It loads NMNIST data, defines a simple MLP,
and trains for a few epochs.

Usage:
    python3 minimal_test.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP for testing"""
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def load_data(data_path='./scratch/data'):
    """Load NMNIST data from cached tensors"""
    data_dir = Path(data_path)

    train_file = data_dir / 'NMNIST_train_data.pt'
    test_file = data_dir / 'NMNIST_test_data.pt'

    if not train_file.exists() or not test_file.exists():
        # Try alternative location
        data_dir = Path('./project_storage/data')
        train_file = data_dir / 'NMNIST_train_data.pt'
        test_file = data_dir / 'NMNIST_test_data.pt'

        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(
                "NMNIST data not found. Please run prepare_data.py first.\n"
                f"Checked: ./scratch/data/ and ./project_storage/data/"
            )

    print(f"Loading data from: {train_file.parent}")
    train_data = torch.load(train_file, map_location='cpu', weights_only=True)
    test_data = torch.load(test_file, map_location='cpu', weights_only=True)

    # Convert boolean to float
    train_x = train_data['data'].float()
    train_y = train_data['labels']
    test_x = test_data['data'].float()
    test_y = test_data['labels']

    print(f"Train data: {train_x.shape}, dtype: {train_x.dtype}, range: [{train_x.min():.3f}, {train_x.max():.3f}]")
    print(f"Test data: {test_x.shape}, dtype: {test_x.dtype}")
    print(f"Number of classes: {len(torch.unique(train_y))}")

    return train_x, train_y, test_x, test_y


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

        if batch_idx % 250 == 0:
            accuracy = 100. * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def main():
    # Configuration
    batch_size = 64
    hidden_size = 512
    learning_rate = 1e-3
    num_epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("Minimal NMNIST Training Test")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print()

    # Load data
    train_x, train_y, test_x, test_y = load_data()

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model parameters
    input_size = train_x.shape[1] * train_x.shape[2] * train_x.shape[3]  # C * H * W
    num_classes = len(torch.unique(train_y))

    print(f"Model: SimpleMLP(input={input_size}, hidden={hidden_size}, classes={num_classes})")
    print()

    # Create model
    model = SimpleMLP(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    print()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        print()

    print("="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    if test_acc > 50:
        print("✓ SUCCESS: Model is learning properly (>50% accuracy)")
    elif test_acc > 20:
        print("⚠ PARTIAL: Model learning but may need tuning (>20% accuracy)")
    else:
        print("✗ FAILURE: Model not learning (≤20% accuracy, close to random)")


if __name__ == '__main__':
    main()
