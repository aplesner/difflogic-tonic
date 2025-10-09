#!/usr/bin/env python3
"""
Layered FeedForward model for DiffLUT experiments.
Based on simple_mnist_test.py structure.
"""

import torch
import torch.nn as nn
from difflut import REGISTRY
from difflut.utils.modules import GroupSum

from models import register_model


@register_model("LayeredFeedForward")
class LayeredFeedForward(nn.Module):
    """
    Multi-layer feedforward network using DiffLUT nodes.
    
    This model mirrors the SimpleDiffLUTModel from simple_mnist_test.py
    but is configurable via config files.
    """
    
    def __init__(self, config: dict, input_size: int, num_classes: int):
        """
        Initialize the LayeredFeedForward model.
        
        Args:
            config: Model configuration dictionary
            input_size: Size of input features (after encoding)
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.config = config
        self.input_size = input_size  # Raw input size (before encoding)
        self.num_classes = num_classes
        
        # Setup encoder
        encoder_config = config.get('encoder', {})
        encoder_name = encoder_config.get('name', 'thermometer')
        encoder_params = encoder_config.get('parameters', {})
        
        encoder_class = REGISTRY.get_encoder(encoder_name)
        self.encoder = encoder_class(**encoder_params)
        self.encoder_fitted = False
        
        # Store layer config for later initialization
        # Layers will be built after encoder is fitted
        self.layers = nn.ModuleList()
        self.layer_config_stored = config
        
        # Output layer (GroupSum)
        self.output_layer = GroupSum(k=num_classes, tau=1)
    
    def _build_node_kwargs(self, node_type: str, node_params: dict, n: int) -> dict:
        """
        Build node-specific kwargs based on node type.
        
        Args:
            node_type: Type of node
            node_params: Parameters from config
            n: Number of inputs
        
        Returns:
            Dictionary of kwargs for node initialization
        """
        node_kwargs = {}
        
        if node_type == 'dwn':
            # DWN-specific parameters
            alpha = node_params.get('alpha', 0.5 * 0.75 ** (n - 1))
            beta = node_params.get('beta', 0.25 / 0.75)
            output_dim = node_params.get('output_dim', 1)
            
            node_kwargs = {
                'alpha': alpha,
                'beta': beta,
                'output_dim': output_dim
            }
        elif node_type in ['probabilistic', 'unbound_probabilistic']:
            # Probabilistic node parameters
            output_dim = node_params.get('output_dim', 1)
            node_kwargs = {
                'output_dim': output_dim
            }
        
        # Add common parameters
        if 'init_fn' in node_params:
            node_kwargs['init_fn'] = node_params['init_fn']
        
        return node_kwargs
    
    def fit_encoder(self, data: torch.Tensor):
        """
        Fit the encoder on training data and build layers.
        
        Args:
            data: Training data tensor (N, input_size)
        """
        if self.encoder_fitted:
            print("Warning: Encoder already fitted, skipping...")
            return
        
        # Fit encoder
        print(f"Fitting encoder on {len(data)} samples...")
        self.encoder.fit(data)
        
        # Get encoded size by encoding a sample
        sample_encoded = self.encoder.encode(data[:1])
        self.encoded_input_size = sample_encoded.shape[1]
        print(f"Encoded input size: {self.encoded_input_size}")
        
        # Now build layers with correct input size
        self._build_layers()
        
        self.encoder_fitted = True
    
    def _build_layers(self):
        """Build layers after encoder is fitted."""
        config = self.layer_config_stored
        current_size = self.encoded_input_size
        
        # Check if using flattened config (easier for Hydra overrides)
        if 'layer_type' in config and 'node_type' in config:
            # Flattened config
            layer_type = config.get('layer_type', 'random')
            hidden_sizes = config.get('hidden_sizes', [1000, 1000])
            node_type_name = config.get('node_type', 'linear_lut')
            n = config.get('num_inputs', 6)
            
            # Get classes from registry
            node_class = REGISTRY.get_node(node_type_name)
            layer_class = REGISTRY.get_layer(layer_type)
            
            # Build node kwargs
            node_kwargs = self._build_node_kwargs(node_type_name, {}, n)
            
            # Create layers for each hidden size
            for hidden_size in hidden_sizes:
                layer = layer_class(
                    input_size=current_size,
                    output_size=hidden_size,
                    node_type=node_class,
                    n=n,
                    node_kwargs=node_kwargs
                )
                self.layers.append(layer)
                current_size = hidden_size
        else:
            # Nested config (backward compatibility)
            layers_config = config.get('layers', [])
            if not layers_config:
                raise ValueError("No layers specified in model config")
            
            for layer_config in layers_config:
                layer_type = layer_config.get('type', 'random')
                hidden_sizes = layer_config.get('hidden_sizes', [1000])
                node_config = layer_config.get('node', {})
                
                node_type_name = node_config.get('type', 'linear_lut')
                node_params = node_config.get('parameters', {})
                
                # Get classes from registry
                node_class = REGISTRY.get_node(node_type_name)
                layer_class = REGISTRY.get_layer(layer_type)
                
                # Extract parameters
                n = node_params.get('num_inputs', 6)
                
                # Build node kwargs based on node type
                node_kwargs = self._build_node_kwargs(node_type_name, node_params, n)
                
                # Create layers for each hidden size
                for hidden_size in hidden_sizes:
                    layer = layer_class(
                        input_size=current_size,
                        output_size=hidden_size,
                        node_type=node_class,
                        n=n,
                        node_kwargs=node_kwargs
                    )
                    self.layers.append(layer)
                    current_size = hidden_size
        
        print(f"Built {len(self.layers)} layers")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input data.
        
        Args:
            x: Input tensor (N, input_size)
        
        Returns:
            Encoded tensor (N, encoded_size)
        """
        return self.encoder.encode(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, input_size) - raw input, will be encoded
        
        Returns:
            Output logits (N, num_classes)
        """
        if not self.encoder_fitted:
            raise RuntimeError("Encoder must be fitted before forward pass. Call fit_encoder() first.")
        
        # Move encoder to same device as input
        self.encoder.to(x.device)
        
        # Encode input
        x = self.encoder.encode(x)
        
        # Clamp to valid range
        x = torch.clamp(x, 0, 1)
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss from all layers.
        
        Returns:
            Total regularization loss
        """
        reg_loss = 0.0
        
        for layer in self.layers:
            if hasattr(layer, 'get_regularization_loss'):
                reg_loss += layer.get_regularization_loss()
        
        return reg_loss
    
    def count_parameters(self) -> dict:
        """
        Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }


def build_model(config: dict, input_size: int, num_classes: int) -> LayeredFeedForward:
    """
    Factory function to build model from config.
    
    Args:
        config: Model configuration dictionary
        input_size: Size of input features
        num_classes: Number of output classes
    
    Returns:
        Initialized model
    """
    model_name = config.get('name', 'LayeredFeedForward')
    
    if model_name == 'LayeredFeedForward':
        return LayeredFeedForward(config.get('params', {}), input_size, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
