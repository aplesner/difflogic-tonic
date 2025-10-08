import os
from pathlib import Path
from omegaconf import OmegaConf
from pydantic import BaseModel, field_validator, Field, model_validator
from typing import Literal, Optional


class PrepareDataConfig(BaseModel):
    """Configuration for data preparation

    This config is specifically for the data preparation pipeline.
    It contains only the fields needed for dataset processing and caching.
    """
    # Dataset settings
    name: str = "NMNIST"  # NMNIST or CIFAR10DVS
    data_root: str = "data/"  # Raw dataset location

    # Frame slicing mode and parameters
    frame_mode: Literal["event_count", "time_window"] = "event_count"

    # Event count mode parameters (used when frame_mode="event_count")
    events_per_frame: Optional[int] = 5000

    # Time window mode parameters (used when frame_mode="time_window")
    time_window: Optional[int] = None  # in microseconds

    # Common parameters
    overlap: float = Field(default=0.75, ge=0.0, le=1.0)  # Overlap between consecutive frames (0.0 to 1.0)
    denoise_time: int = 1000

    # Processing settings
    num_threads: int = 6

    # Cache settings
    reset_cache: bool = False  # Force regeneration
    output_suffix: Optional[str] = None  # Optional suffix for output files

    # Seed for reproducibility
    seed: int = 42

    # Testing / validation settings
    test_split: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_frame_mode_parameters(self):
        """Ensure the correct parameters are set based on frame_mode"""
        if self.frame_mode == "event_count":
            if self.events_per_frame is None:
                raise ValueError("events_per_frame must be set when frame_mode='event_count'")
        elif self.frame_mode == "time_window":
            if self.time_window is None:
                raise ValueError("time_window must be set when frame_mode='time_window'")
        return self

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
            'frame_mode': data_section.get('frame_mode', 'event_count'),
            'events_per_frame': data_section.get('events_per_frame'),
            'time_window': data_section.get('time_window'),
            'overlap': data_section.get('overlap', 0),
            'denoise_time': data_section.get('denoise_time', 1000),
            'num_threads': data_section.get('num_threads', 6),
            'test_split': data_section.get('test_split', 0.2),
            'reset_cache': data_section.get('reset_cache', False),
            'output_suffix': data_section.get('output_suffix'),
            'seed': base_section.get('seed', 42),
        }

        return cls(**prep_config)

    def get_cache_identifier(self) -> str:
        """Generate a unique identifier for this configuration's cached output"""
        if self.output_suffix:
            return self.output_suffix

        if self.frame_mode == "event_count":
            base = f"events_{self.events_per_frame}_overlap{self.overlap}"
        else:  # time_window
            base = f"time_{self.time_window}_overlap{self.overlap}"

        if self.denoise_time > 0:
            base += f"_denoise{self.denoise_time}"

        return base