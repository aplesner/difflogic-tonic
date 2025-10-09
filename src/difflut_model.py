import torch
import torch.nn as nn


from difflut.difflut import REGISTRY
from difflut.difflut.utils.modules import GroupSum




class DiffLUTConfig:
    def __init__(
            self,
            nodetypes: dict,
            layertypes: dict,
            n: int = 6,
            num_layers: int = 2,
            default_layer: str = "learnable",
            default_node: str = "neurallut",
            default_hidden_size: int = 2048,
            layer_overrides: dict = {},
        ):
        self.nodetypes = nodetypes
        self.layertypes = layertypes

        self.n = n

        self.default_layer = default_layer
        self.default_node = default_node
        self.default_hidden_size = default_hidden_size

        self.num_layers = num_layers

        self.layer_overrides = layer_overrides

    def create(self, input_size, num_classes, device):
        node_regularizers = None  # No regularizers by default

        layer_configs = [
            {
                'node': {**self.nodetypes[self.layer_overrides.get(i, {}).get('node', self.default_node)]},
                'layer': {**self.layertypes[self.layer_overrides.get(i, {}).get('layer', self.default_layer)]},
                'hidden_size': self.layer_overrides.get(i, {}).get('hidden_size', self.default_hidden_size)
            } for i in range(self.num_layers)
        ]

        return DiffLUTModel(
            input_size=input_size,
            output_size=num_classes,
            n=self.n,
            layer_configs=layer_configs,
            node_regularizers=node_regularizers,
        ).to(device)


class DiffLUTModel(nn.Module):
    """
    DiffLUT Model built using registry components.
    """
    def __init__(self, input_size, output_size=10, n=6, layer_configs=[], node_regularizers=None):
        super().__init__()

        self.layer_configs = layer_configs
        self.node_regularizers = node_regularizers

        # Get node and layer classes from registry
        print(f"\nBuilding model with:")
        for i, config in enumerate(layer_configs):
            print(f"  Layer {i+1}:")
            print(f"    Node config: {config['node']}")
            print(f"    Layer config: {config['layer']}")
            print(f"    Hidden size (nodes): {config['hidden_size']}")
            print(f"    LUT inputs (n): {n}")
        if node_regularizers:
            print(f"  Regularizers: {list(node_regularizers.keys())}")

        # self.node_class = REGISTRY.get_node(node_config._target_)
        # self.layer_class = REGISTRY.get_layer(layer_config._target_)
        self.n = n
        
        # Build layers
        layers = []
        prev_size = input_size
        
        # for i, hidden_size in enumerate(hidden_sizes):
        for i, config in enumerate(layer_configs):
            hidden_size = config['hidden_size']
            hidden_size = (hidden_size // output_size) * output_size
            # Configure node parameters based on type
            # node_kwargs = {}
            node_kwargs = config['node']
            node_type = node_kwargs.pop('node_type')
            node_class = REGISTRY.get_node(node_type)

            layer_kwargs = config['layer']
            layer_type = layer_kwargs.pop('layer_type')
            layer_class = REGISTRY.get_layer(layer_type)

            if node_type == 'dwn':
                # DWN-specific parameters for gradient computation
                node_kwargs.update({
                    'alpha': 0.5 * 0.75 ** (n - 1), 
                    'beta': 0.25 / 0.75,
                    'output_dim': 1
                })

            # Add regularizers if specified
            if node_regularizers:
                node_kwargs['regularizers'] = node_regularizers
            
            # Create layer using registry components
            # layer = self.layer_class(
            layer = layer_class(
                input_size=prev_size,
                output_size=hidden_size,
                # node_type=self.node_class,
                node_type=node_class,
                n=n,
                node_kwargs=node_kwargs
            )
            layers.append(layer)
            
            print(f"  Layer {i+1}: {prev_size} → {hidden_size} (nodes: {hidden_size})")
            prev_size = hidden_size
        
        # Stack all layers
        self.layers = nn.ModuleList(layers)
        
        # Output layer: group nodes and sum for classification
        if prev_size % output_size != 0:
            raise ValueError(f"Last hidden size {prev_size} must be divisible by output size {output_size}")
        

        tau = hidden_size // output_size
        self.output_layer = GroupSum(k=output_size, tau=tau)
        print(f"  Output layer: {prev_size} → {output_size} (via GroupSum)")
    
    def forward(self, x):
        """Forward pass through all layers."""
        # Ensure inputs are in [0, 1] range
        x = torch.clamp(x, 0, 1)
        # Pass through each LUT layer
        for i, layer in enumerate(self.layers):
            x = layer(x)

        # Group and sum for final output
        x = self.output_layer(x)
        return x
    
    def regularization(self):
        """Compute regularization loss across all layers."""
        reg = 0.0
        for layer in self.layers:
            if hasattr(layer, 'regularization'):
                layer_reg = layer.regularization()
                # Only add if it has gradients or is non-zero
                if isinstance(layer_reg, torch.Tensor):
                    reg = reg + layer_reg
                else:
                    reg = reg + layer_reg
        return reg
