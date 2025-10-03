import os
from pathlib import Path
from omegaconf import OmegaConf
from pydantic import BaseModel, field_validator, Field


class PrepareDataConfig(BaseModel):
    """Configuration for data preparation

    This config is specifically for the data preparation pipeline.
    It contains only the fields needed for dataset processing and caching.
    """
    # Dataset settings
    name: str = "NMNIST"  # NMNIST or CIFAR10DVS
    data_root: str = "data/"  # Raw dataset location

    # Preprocessing parameters
    events_per_frame: int = 5000
    overlap: int = 0
    denoise_time: int = 1000

    # Processing settings
    num_threads: int = 6

    # Cache settings
    reset_cache: bool = False  # Force regeneration

    # Seed for reproducibility
    seed: int = 42

    # Testing / validation settings
    test_split: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator('data_root')
    @classmethod
    def resolve_data_root(cls, v: str) -> str:
        """Resolve data root path, using SCRATCH_STORAGE_DIR if available"""
        scratch_dir = os.environ.get('SCRATCH_STORAGE_DIR')
        if scratch_dir:
            return str(Path(scratch_dir) / v)
        return v

    @classmethod
    def from_yaml(cls, path: str):
        """Load config from YAML file, extracting only preparation-relevant fields"""
        conf = OmegaConf.load(path)
        config_dict = OmegaConf.to_container(conf, resolve=True)

        if not isinstance(config_dict, dict):
            raise ValueError("Configuration file must contain a dictionary at the top level.")

        # Extract data section (where most prep config lives)
        data_section = config_dict.get('data', {})
        base_section = config_dict.get('base', {})

        # Build preparation config from relevant fields
        prep_config = {
            'name': data_section.get('name', 'NMNIST'),
            'data_root': data_section.get('data_root', 'data/'),
            'events_per_frame': data_section.get('events_per_frame', 5000),
            'overlap': data_section.get('overlap', 0),
            'denoise_time': data_section.get('denoise_time', 1000),
            'num_threads': data_section.get('num_threads', 6),
            'test_split': data_section.get('test_split', 0.2),
            'reset_cache': data_section.get('reset_cache', False),
            'seed': base_section.get('seed', 42),
        }

        return cls(**prep_config)