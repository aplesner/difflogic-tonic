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


def get_model_input_shape(data_config: config.DataConfig):
    """Get input shape from sensor size (Note: SensorSizes are NHWC, but we use NCHW)"""
    sensor_size = data.SensorSizes[data_config.name].value  # (H, W, C) format
    # Convert to NCHW: (C, H, W)
    return (sensor_size[2], sensor_size[0], sensor_size[1])


def get_num_classes(data_config: config.DataConfig):
    """Get number of output classes based on dataset"""
    return data.OutputClasses[data_config.name].value