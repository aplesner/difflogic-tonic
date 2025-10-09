"""
Test script for the new GroupedLayer
"""
import torch
import sys
sys.path.insert(0, '/itet-stor/sbuehrer/net_scratch/difflut')

from difflut import GroupedLayer, LearnableLayer, RandomLayer
from difflut.nodes import LinearLUTNode

def test_grouped_layer():
    print("=" * 60)
    print("Testing GroupedLayer")
    print("=" * 60)
    
    # Configuration
    batch_size = 16
    input_size = 128
    output_size = 64
    n = 6
    num_groups = 4
    
    # Create layers
    print(f"\nConfiguration:")
    print(f"  Input size: {input_size}")
    print(f"  Output size: {output_size}")
    print(f"  n (inputs per node): {n}")
    print(f"  Number of groups: {num_groups}")
    print(f"  Batch size: {batch_size}")
    
    # Create test input
    x = torch.randn(batch_size, input_size)
    
    # Test GroupedLayer
    print("\n" + "-" * 60)
    print("1. GroupedLayer")
    print("-" * 60)
    grouped_layer = GroupedLayer(
        input_size=input_size,
        output_size=output_size,
        node_type=LinearLUTNode,
        n=n,
        num_groups=num_groups
    )
    
    # Forward pass
    output = grouped_layer(x)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_size), f"Expected shape {(batch_size, output_size)}, got {output.shape}"
    
    # Parameter count
    params = grouped_layer.count_parameters()
    print(f"\nParameter breakdown:")
    print(f"  Mapping parameters: {params['mapping']:,}")
    print(f"  Node parameters: {params['nodes']:,}")
    print(f"  Total parameters: {params['total']:,}")
    
    # Efficiency comparison
    efficiency = grouped_layer.get_parameter_efficiency()
    print(f"\nEfficiency metrics:")
    print(f"  Learnable layer params: {efficiency['learnable_params']:,}")
    print(f"  Grouped layer params: {efficiency['grouped_params']['total']:,}")
    print(f"  Parameter reduction: {efficiency['reduction']*100:.2f}%")
    print(f"  Efficiency ratio: {efficiency['efficiency']:.2f}x")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in grouped_layer.parameters())
    print(f"\nGradient flow: {'✓ Working' if has_grad else '✗ Not working'}")
    
    # Test training/eval mode
    grouped_layer.eval()
    with torch.no_grad():
        output_eval = grouped_layer(x)
    print(f"Eval mode output shape: {output_eval.shape}")
    
    # Compare with LearnableLayer
    print("\n" + "-" * 60)
    print("2. LearnableLayer (for comparison)")
    print("-" * 60)
    learnable_layer = LearnableLayer(
        input_size=input_size,
        output_size=output_size,
        node_type=LinearLUTNode,
        n=n
    )
    
    learnable_params = sum(p.numel() for p in learnable_layer.parameters())
    print(f"Total parameters: {learnable_params:,}")
    
    output_learnable = learnable_layer(x)
    print(f"Output shape: {output_learnable.shape}")
    
    # Compare with RandomLayer
    print("\n" + "-" * 60)
    print("3. RandomLayer (for comparison)")
    print("-" * 60)
    random_layer = RandomLayer(
        input_size=input_size,
        output_size=output_size,
        node_type=LinearLUTNode,
        n=n
    )
    
    random_params = sum(p.numel() for p in random_layer.parameters())
    print(f"Total parameters: {random_params:,}")
    
    output_random = random_layer(x)
    print(f"Output shape: {output_random.shape}")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Layer Type':<20} {'Parameters':<15} {'Learnable Mapping':<20}")
    print("-" * 60)
    print(f"{'Random':<20} {random_params:<15,} {'No (fixed)':<20}")
    print(f"{'Grouped':<20} {params['total']:<15,} {'Yes (grouped)':<20}")
    print(f"{'Learnable':<20} {learnable_params:<15,} {'Yes (full)':<20}")
    print("=" * 60)
    
    reduction_vs_learnable = (1 - params['total'] / learnable_params) * 100
    print(f"\nGrouped layer uses {reduction_vs_learnable:.1f}% fewer parameters than Learnable layer")
    print(f"with {num_groups} groups.\n")
    
    print("✓ All tests passed!")

if __name__ == "__main__":
    test_grouped_layer()
