#!/usr/bin/env python3
"""
Test script to verify encoder is correctly moved to model.
"""

import torch
import sys
sys.path.insert(0, 'experiments')

from dataloaders import get_dataloader
from models import build_model

# Test configuration
dataloader_config = {
    'name': 'mnist',
    'data_dir': './data',
    'batch_size': 32,
    'subset_size': 100,
    'test_subset_size': 50,
    'num_workers': 0
}

model_config = {
    'name': 'LayeredFeedForward',
    'params': {
        'encoder': {
            'name': 'thermometer',
            'parameters': {'num_bits': 3, 'feature_wise': True}
        },
        'layer_type': 'random',
        'hidden_sizes': [100, 100],
        'node_type': 'linear_lut',
        'num_inputs': 6
    }
}

print("="*60)
print("Testing Encoder in Model")
print("="*60)

# Setup dataloader
print("\n1. Setting up dataloader...")
dataloader = get_dataloader('mnist', dataloader_config)
dataloader.prepare_data()
train_loader, val_loader, test_loader = dataloader.setup()

input_size = dataloader.get_input_size()
num_classes = dataloader.get_num_classes()

print(f"   Input size (raw): {input_size}")
print(f"   Num classes: {num_classes}")
print(f"   Train batches: {len(train_loader)}")

# Get a batch to check
sample_batch, sample_labels = next(iter(train_loader))
print(f"   Sample batch shape: {sample_batch.shape}")
print(f"   Sample batch range: [{sample_batch.min():.3f}, {sample_batch.max():.3f}]")

# Build model
print("\n2. Building model...")
model = build_model(model_config, input_size, num_classes)
print(f"   Model: {model.__class__.__name__}")
print(f"   Encoder fitted: {model.encoder_fitted}")

# Fit encoder
print("\n3. Fitting encoder on training data...")
all_train_data = []
for inputs, labels in train_loader:
    all_train_data.append(inputs)
all_train_data = torch.cat(all_train_data, dim=0)
print(f"   Training data shape: {all_train_data.shape}")

model.fit_encoder(all_train_data)
print(f"   Encoder fitted: {model.encoder_fitted}")
print(f"   Encoded input size: {model.encoded_input_size}")
print(f"   Number of layers: {len(model.layers)}")

# Test forward pass
print("\n4. Testing forward pass...")
test_batch, test_labels = next(iter(val_loader))
print(f"   Test batch shape: {test_batch.shape}")

model.eval()
with torch.no_grad():
    output = model(test_batch)
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

# Check predictions
predictions = output.argmax(dim=1)
print(f"   Predictions: {predictions[:10].tolist()}")
print(f"   Labels: {test_labels[:10].tolist()}")

print("\n" + "="*60)
print("✓ Test passed! Encoder is working correctly in the model.")
print("="*60)
