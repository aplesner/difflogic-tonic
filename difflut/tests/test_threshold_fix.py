#!/usr/bin/env python3
"""
Test script to verify DWN and Hybrid nodes work correctly with [0, 1] inputs.
This tests the threshold fix from 0.0 to 0.5.
"""

import torch
import sys
from pathlib import Path

# Add difflut to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from difflut.nodes import DWNNode, HybridNode, ProbabilisticNode

def test_node_with_thermometer_inputs():
    """Test nodes with thermometer-style [0, 1] inputs."""
    
    print("="*60)
    print("Testing DWN and Hybrid nodes with [0, 1] inputs")
    print("="*60)
    
    # Create sample thermometer-encoded inputs
    # Thermometer encoding produces values in [0, 1]
    batch_size = 4
    num_inputs = 6
    
    # Create diverse test inputs
    x_test = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # All zeros
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # All ones
        [0.2, 0.4, 0.6, 0.8, 0.9, 1.0],  # Thermometer-like gradient
        [0.1, 0.3, 0.5, 0.7, 0.9, 0.95], # Another gradient
    ])
    
    print(f"\nTest input shape: {x_test.shape}")
    print(f"Test input range: [{x_test.min():.2f}, {x_test.max():.2f}]")
    print("\nTest inputs:")
    print(x_test)
    
    # Test DWN node
    print("\n" + "-"*60)
    print("Testing DWN Node")
    print("-"*60)
    
    dwn = DWNNode(num_inputs=num_inputs, output_dim=1, use_cuda=False)
    dwn.eval()
    
    with torch.no_grad():
        dwn_output = dwn(x_test)
    
    print(f"DWN output shape: {dwn_output.shape}")
    print(f"DWN output: {dwn_output}")
    print(f"DWN output range: [{dwn_output.min():.4f}, {dwn_output.max():.4f}]")
    print(f"DWN output unique values: {len(dwn_output.unique())}")
    
    # Check that different inputs give different outputs
    unique_outputs = len(dwn_output.unique())
    if unique_outputs > 1:
        print("✓ DWN produces varied outputs (good!)")
    else:
        print("✗ WARNING: DWN produces identical outputs (may indicate threshold issue)")
    
    # Test gradient flow
    dwn.train()
    x_train = x_test.clone().requires_grad_(True)
    output = dwn(x_train)
    loss = output.sum()
    loss.backward()
    
    if x_train.grad is not None and x_train.grad.abs().sum() > 0:
        print(f"✓ DWN gradients flow (sum: {x_train.grad.abs().sum():.4f})")
    else:
        print("✗ WARNING: DWN gradients are zero")
    
    # Test Hybrid node
    print("\n" + "-"*60)
    print("Testing Hybrid Node")
    print("-"*60)
    
    hybrid = HybridNode(num_inputs=num_inputs, output_dim=1, use_cuda=False)
    hybrid.eval()
    
    with torch.no_grad():
        hybrid_output = hybrid(x_test)
    
    print(f"Hybrid output shape: {hybrid_output.shape}")
    print(f"Hybrid output: {hybrid_output}")
    print(f"Hybrid output range: [{hybrid_output.min():.4f}, {hybrid_output.max():.4f}]")
    print(f"Hybrid output unique values: {len(hybrid_output.unique())}")
    
    # Check that different inputs give different outputs
    unique_outputs = len(hybrid_output.unique())
    if unique_outputs > 1:
        print("✓ Hybrid produces varied outputs (good!)")
    else:
        print("✗ WARNING: Hybrid produces identical outputs (may indicate threshold issue)")
    
    # Test gradient flow
    hybrid.train()
    x_train = x_test.clone().requires_grad_(True)
    output = hybrid(x_train)
    loss = output.sum()
    loss.backward()
    
    if x_train.grad is not None and x_train.grad.abs().sum() > 0:
        print(f"✓ Hybrid gradients flow (sum: {x_train.grad.abs().sum():.4f})")
    else:
        print("✗ WARNING: Hybrid gradients are zero")
    
    # Test Probabilistic node for comparison
    print("\n" + "-"*60)
    print("Testing Probabilistic Node (for comparison)")
    print("-"*60)
    
    prob = ProbabilisticNode(num_inputs=num_inputs, output_dim=1)
    prob.eval()
    
    with torch.no_grad():
        prob_output = prob(x_test)
    
    print(f"Probabilistic output shape: {prob_output.shape}")
    print(f"Probabilistic output: {prob_output}")
    print(f"Probabilistic output range: [{prob_output.min():.4f}, {prob_output.max():.4f}]")
    
    # Test gradient flow
    prob.train()
    x_train = x_test.clone().requires_grad_(True)
    output = prob(x_train)
    loss = output.sum()
    loss.backward()
    
    if x_train.grad is not None and x_train.grad.abs().sum() > 0:
        print(f"✓ Probabilistic gradients flow (sum: {x_train.grad.abs().sum():.4f})")
    else:
        print("✗ WARNING: Probabilistic gradients are zero")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


if __name__ == "__main__":
    test_node_with_thermometer_inputs()
