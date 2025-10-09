import os
from pathlib import Path
import logging

logger = logging.getLogger()

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator
import torch

from .model_config import ModelConfig


class WandBConfig(BaseModel):
    online: bool = True
    project: str = "difflogic-tonic"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = []
    notes: str | None = None


class BaseConfig(BaseModel):
    seed: int = 42
    debug: bool = False
    job_id: str = "default"
    wandb: WandBConfig = WandBConfig()

    @field_validator('job_id')
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        return v if v else "default"


class AugmentationConfig(BaseModel):
    """Configuration for data augmentation (applied only to training data)"""
    # Flip augmentations
    horizontal_flip_probability: float = Field(default=0.0, ge=0.0, le=1.0)  # probability for horizontal flip
    vertical_flip_probability: float = Field(default=0.0, ge=0.0, le=1.0)  # probability for vertical flip

    # Crop augmentation
    random_crop_padding: int = 0  # padding for random crop (0 = disabled)

    # Noise augmentation (custom for spike data)
    salt_pepper_noise: float = Field(default=0.0, ge=0.0, le=1.0)  # probability [0.0, 1.0] for flipping random pixels


class DataConfig(BaseModel):
    name: str = "NMNIST"
    cache_identifier: str | None = None  # Specify which cached variant to use (e.g., "events_20000_overlap0_denoise5000")
    downsample_pool_size: int | None = None  # e.g., 2 for 2x2 max pooling, 4 for 4x4
    augmentation: AugmentationConfig = AugmentationConfig()


class DataLoaderConfig(BaseModel):
    batch_size: int = 64
    num_workers: int = 6
    prefetch_factor: int = 5
    pin_memory: bool = True
    shuffle_train: bool = True


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration"""
    enabled: bool = False
    type: str = "step"  # options: step, cosine, reduce_on_plateau
    step_size: int = 10  # for step scheduler
    gamma: float = 0.1  # for step scheduler
    min_lr: float = 1e-6  # for cosine and reduce_on_plateau
    factor: float = 0.5  # for reduce_on_plateau
    patience: int = 5  # for reduce_on_plateau


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration"""
    enabled: bool = False
    monitor: str = "val/acc"  # metric to monitor
    patience: int = 10  # number of epochs with no improvement
    mode: str = "max"  # max for accuracy, min for loss
    min_delta: float = 0.0  # minimum change to qualify as improvement


class LightningConfig(BaseModel):
    """PyTorch Lightning trainer configuration"""
    enable_progress_bar: bool = True
    log_every_n_steps: int = 50
    gradient_clip_val: float | None = None  # gradient clipping value (None = disabled)
    accumulate_grad_batches: int = 1  # gradient accumulation
    check_val_every_n_epoch: int = 1  # validation frequency
    num_sanity_val_steps: int = 2  # sanity check validation steps before training


class TrainConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    epochs: int = 3
    learning_rate: float = 1e-3
    log_interval: int = 250
    debugging_steps: int = 10
    save_model: bool = False
    checkpoint_interval_minutes: float = 10.0  # time-based checkpointing (deprecated with Lightning)
    model_path: str = "models/"
    device: str | torch.device = "cuda"
    dtype: str | torch.dtype = "float16"  # options: float16, bfloat16, float32
    dataloader: DataLoaderConfig = DataLoaderConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    lightning: LightningConfig = LightningConfig()

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
    
    @field_validator('dtype')
    @classmethod
    def validate_dtype(cls, v: str | torch.dtype) -> torch.dtype:
        if isinstance(v, str):
            dtype_map = {
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'float32': torch.float32
            }
            if v in dtype_map:
                return dtype_map[v]
            else:
                raise ValueError(f"Unsupported dtype string: {v}. Supported: {list(dtype_map.keys())}")
        elif isinstance(v, torch.dtype):
            if v in {torch.float16, torch.bfloat16, torch.float32}:
                return v
            else:
                raise ValueError(f"Unsupported torch.dtype: {v}. Supported: float16, bfloat16, float32")
        else:
            raise TypeError(f"dtype must be a string or torch.dtype, got {type(v)}")

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
    