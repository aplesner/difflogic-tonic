#!/usr/bin/env python3
"""Debug script to test if models work with the actual data loading pipeline"""

import torch
import torch.nn as nn
import torch.optim as optim

# Define models locally without importing
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.network(x)


class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        channels, height, width = input_shape

        self.conv1 = nn.Conv2d(channels, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        conv_output_height = height // 4
        conv_output_width = width // 4
        flattened_size = 64 * conv_output_height * conv_output_width

        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def test_model(model_name, model, data_batch, labels_batch, device):
    """Test a model with a batch of data"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Test forward pass
    print(f"Input shape: {data_batch.shape}, dtype: {data_batch.dtype}")
    if data_batch.dtype == torch.bool:
        display_batch = data_batch.float()
    else:
        display_batch = data_batch
    print(f"Input range: [{display_batch.min():.3f}, {display_batch.max():.3f}]")
    print(f"Input mean: {display_batch.mean():.3f}, std: {display_batch.std():.3f}")

    model.train()
    losses = []
    accuracies = []

    for step in range(10):
        # Convert bool to float if needed
        if data_batch.dtype == torch.bool:
            data = data_batch.float()
        else:
            data = data_batch

        data = data.to(device)
        labels = labels_batch.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()

        # Check for NaN gradients
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"  WARNING: NaN gradient in {name}")
                has_nan = True

        if not has_nan:
            optimizer.step()

        # Calculate accuracy
        pred = output.argmax(dim=1)
        acc = (pred == labels).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc * 100)

        if step % 2 == 0:
            print(f"  Step {step}: Loss={loss.item():.4f}, Acc={acc*100:.2f}%")

    print(f"\nFinal metrics:")
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (change: {losses[-1]-losses[0]:.4f})")
    print(f"  Accuracy: {accuracies[0]:.2f}% -> {accuracies[-1]:.2f}%")

    if losses[-1] < losses[0]:
        print(f"  ✓ Loss is decreasing")
    else:
        print(f"  ✗ Loss is NOT decreasing!")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load actual data
    print("\nLoading NMNIST data...")
    train_data = torch.load('./scratch/data/NMNIST_train_data.pt', map_location='cpu', weights_only=True)

    # Take first batch
    batch_size = 64
    data_batch = train_data['data'][:batch_size]
    labels_batch = train_data['labels'][:batch_size]

    print(f"Data shape: {data_batch.shape}")
    print(f"Data dtype: {data_batch.dtype}")
    print(f"Labels: {labels_batch[:10]}")

    # Model parameters
    input_shape = (2, 34, 34)  # NMNIST: 2 channels (polarities), 34x34
    input_size = 2 * 34 * 34
    num_classes = 10

    # Test MLP
    mlp = SimpleMLP(input_size, hidden_size=512, num_classes=num_classes)
    test_model("MLP", mlp, data_batch, labels_batch, device)

    # Test CNN
    cnn = SimpleCNN(input_shape, num_classes=num_classes)
    test_model("CNN", cnn, data_batch, labels_batch, device)


if __name__ == '__main__':
    main()
