import random
import logging

import numpy as np
import torch

from . import config
from . import data

logger = logging.getLogger(__name__)


def setup_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug(f"Random seeds set to {seed}")


def get_model_input_shape(cfg: config.Config):
    """Get input shape from sensor size accounting for downsampling

    Note: SensorSizes are NHWC, but we use NCHW format
    """
    sensor_size = data.SensorSizes[cfg.data.name].value  # (H, W, C) format
    height, width, channels = sensor_size

    # Apply downsampling if configured
    if cfg.data.downsample_pool_size:
        height = height // cfg.data.downsample_pool_size
        width = width // cfg.data.downsample_pool_size
        logger.debug(f"Input shape adjusted for pooling size {cfg.data.downsample_pool_size}: ({channels}, {height}, {width})")

    # Convert to NCHW: (C, H, W)
    return (channels, height, width)


def get_num_classes(data_config: config.DataConfig):
    """Get number of output classes based on dataset"""
    return data.OutputClasses[data_config.name].value