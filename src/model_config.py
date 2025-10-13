from pydantic import BaseModel

class MLPConfig(BaseModel):
    hidden_size: int = 512
    num_hidden_layers: int = 6

class CNNConfig(BaseModel):
    conv1_channels: int = 32
    conv2_channels: int = 64
    fc_hidden_size: int = 128

class DiffLogicConfig(BaseModel):
    architecture: str = "fully_connected"  # Options: fully_connected, convolutional
    num_neurons: int = 64_000
    num_layers: int = 4
    connections: str = "random"
    grad_factor: float = 1.0
    tau: float = 10.0

class DiffLUTConfig(BaseModel):
    nodetypes: dict = {}
    layertypes: dict = {}
    n: int = 6
    num_layers: int = 2
    default_layer: str = "learnable"
    default_node: str = "neurallut"
    default_hidden_size: int = 2048
    layer_overrides: dict = {}


class ModelConfig(BaseModel):
    model_type: str = "MLP"
    mlp: MLPConfig = MLPConfig()
    cnn: CNNConfig = CNNConfig()
    difflogic: DiffLogicConfig = DiffLogicConfig()
    difflut: DiffLUTConfig = DiffLUTConfig()
