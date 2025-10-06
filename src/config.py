import os
from pathlib import Path
import logging

logger = logging.getLogger()

from omegaconf import OmegaConf
from pydantic import BaseModel, field_validator
import torch

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


class ModelConfig(BaseModel):
    model_type: str = "MLP"
    mlp: MLPConfig = MLPConfig()
    cnn: CNNConfig = CNNConfig()
    diff_logic: DiffLogicConfig = DiffLogicConfig()

class DataLoaderConfig(BaseModel):
    batch_size: int = 64
    num_workers: int = 6
    prefetch_factor: int = 5
    pin_memory: bool = True
    shuffle_train: bool = True

class DataConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str = "NMNIST"
    data_root: str = "data/"
    metadata_path: str = "metadata/"
    events_per_frame: int = 5000
    overlap: int = 0
    denoise_time: int = 1000
    reset_cache: bool = False
    dataloader: DataLoaderConfig = DataLoaderConfig()

    @field_validator('data_root', 'metadata_path')
    @classmethod
    def resolve_storage_path(cls, v: str) -> str:
        scratch_dir = os.environ.get('SCRATCH_STORAGE_DIR')
        if scratch_dir:
            return str(Path(scratch_dir) / v)
        return v


class TrainConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    epochs: int = 3
    learning_rate: float = 1e-3
    log_interval: int = 1
    debugging_steps: int = 1000
    save_model: bool = False
    checkpoint_interval_minutes: float = 5.0  # time-based checkpointing
    model_path: str = "models/"
    device: str | torch.device = "cpu"
    dataloader: DataLoaderConfig = DataLoaderConfig()

    @field_validator('model_path')
    @classmethod
    def resolve_storage_path(cls, v: str) -> str:
        scratch_dir = os.environ.get('SCRATCH_STORAGE_DIR')
        if scratch_dir:
            return str(Path(scratch_dir) / v)
        return v

    @field_validator('device')
    @classmethod
    def validate_device(cls, v: str | torch.device) -> torch.device:
        if isinstance(v, str):
            v = torch.device(v)

        if v.type == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, switching to CPU.")
            logger.debug(f"Available devices: {torch.cuda.device_count()}")
            return torch.device('cpu')

        return v

class WandBConfig(BaseModel):
    online: bool = False
    project: str = "difflogic-tonic"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = []
    notes: str | None = None

class BaseConfig(BaseModel):
    seed: int = 42
    debug: bool = False
    wandb: WandBConfig = WandBConfig()

class Config(BaseModel):
    base: BaseConfig = BaseConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    
    @classmethod
    def from_yaml(cls, path: str, overrides: list[str] | None = None):
        """Load config from YAML file with optional CLI overrides

        Args:
            path: Path to YAML config file
            overrides: List of dotted-path overrides (e.g., ["train.epochs=10", "model.mlp.hidden_size=512"])
        """
        conf = OmegaConf.load(path)

        # Apply CLI overrides if provided
        if overrides:
            override_conf = OmegaConf.from_dotlist(overrides)
            conf = OmegaConf.merge(conf, override_conf)

        config_dict = OmegaConf.to_container(conf, resolve=True)
        if isinstance(config_dict, dict):
            return cls(**config_dict) # type: ignore
        else:
            raise ValueError("Configuration file must contain a dictionary at the top level.")
    