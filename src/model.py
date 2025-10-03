from difflogic import LogicLayer, GroupSum
import torch
import torch.nn as nn

from .config import ModelConfig, MLPConfig, CNNConfig, DiffLogicConfig


class DiffLogic(nn.Module):
    def __init__(self, config: DiffLogicConfig, input_size: int, num_classes: int, device: torch.device = torch.device('cpu')):
        super().__init__()
        logic_layers = []

        llkw = dict(grad_factor=config.grad_factor, connections=config.connections)

        ####################################################################################################################

        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(
            LogicLayer(
                in_dim=input_size, 
                out_dim=config.num_neurons, 
                connections='random', 
                grad_factor=config.grad_factor, 
                device=device.type
            )
        )
        for _ in range(config.num_layers - 1):
            logic_layers.append(
                LogicLayer(
                    in_dim=config.num_neurons, 
                    out_dim=config.num_neurons, 
                    connections=config.connections,
                    grad_factor=config.grad_factor, 
                    device=device.type
                )
            )

        self.network = torch.nn.Sequential(
            *logic_layers,
            GroupSum(num_classes, config.tau)
        )

    def forward(self, x: torch.Tensor):
        x = self.network(x)
        return x


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron"""

    def __init__(self, config: MLPConfig, input_size: int, num_classes: int):
        super().__init__()

        layers = []
        current_size = input_size

        # Hidden layers
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(current_size, config.hidden_size))
            layers.append(nn.ReLU())
            current_size = config.hidden_size

        # Output layer
        layers.append(nn.Linear(current_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # Flatten input: [batch, channels, height, width] -> [batch, channels*height*width]
        x = x.flatten(start_dim=1)
        return self.network(x)


class CNN(nn.Module):
    """Simple CNN with two convolutional layers"""

    def __init__(self, config: CNNConfig, input_shape: tuple[int, int, int], num_classes: int):
        super().__init__()

        channels, height, width = input_shape

        # First conv layer: kernel_size=5
        self.conv1 = nn.Conv2d(channels, config.conv1_channels, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second conv layer: kernel_size=3
        self.conv2 = nn.Conv2d(config.conv1_channels, config.conv2_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Calculate flattened size after convolutions and pooling
        conv_output_height = height // 4  # Two pooling layers with stride 2
        conv_output_width = width // 4
        flattened_size = config.conv2_channels * conv_output_height * conv_output_width

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, config.fc_hidden_size)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(config.fc_hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        # Conv layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten and FC layers
        x = x.flatten(start_dim=1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(config: ModelConfig, input_shape: tuple, num_classes: int, device: torch.device = torch.device('cpu')) -> nn.Module:
    """Create model based on config"""
    if config.model_type == "MLP":
        # Calculate input size from shape (channels, height, width)
        input_size = input_shape[0] * input_shape[1] * input_shape[2]
        model = MLP(config.mlp, input_size, num_classes)
    elif config.model_type == "CNN":
        model = CNN(config.cnn, input_shape, num_classes)
    elif config.model_type == "DiffLogic":
        input_size = input_shape[0] * input_shape[1] * input_shape[2]
        model = DiffLogic(config=config.diff_logic, input_size=input_size, num_classes=num_classes, device=device)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    return model.to(device)
