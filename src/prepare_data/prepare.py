import os
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Any


import numpy as np
import torch
import tonic
import tonic.transforms as tonic_transforms
import tqdm

from .config import PrepareDataConfig
from ..classes import DatasetSplit, PreparedDataset, SubsetDataset
from ..io_funcs import save_data_splits

logger = logging.getLogger(__name__)


class TransformPolarities:
    """Transform polarities into binary representation."""
    def __call__(self, frames: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return frames > 0


class DatasetProcessor:
    """Picklable processor for multiprocessing

    This class wraps dataset and transform to make them picklable for multiprocessing.Pool.
    Nested functions cannot be pickled, but top-level classes can be.
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __call__(self, i):
        """Process a single dataset item by index"""
        events, label = self.dataset[i]
        frames = self.transform(events)
        return frames, label


def process_dataset(dataset, transform, prep_config: PrepareDataConfig) -> torch.utils.data.TensorDataset:
    """Process dataset using multiprocessing with picklable processor"""
    processor = DatasetProcessor(dataset, transform)

    if prep_config.num_threads < 1:
        num_threads = os.cpu_count() or 1
        logger.info(f"Using all available CPU threads: {num_threads}")
    elif os.cpu_count() and prep_config.num_threads > os.cpu_count():  # type: ignore
        logger.warning(f"Requested {prep_config.num_threads} threads but only {os.cpu_count()} are available.")
        num_threads = os.cpu_count()
    else:
        num_threads = prep_config.num_threads
    logger.info(f"Processing dataset with {num_threads} threads...")

    with Pool(num_threads) as p:
        dataset_size = len(dataset)
        results = list(
            tqdm.tqdm(
                p.imap(
                    processor,
                    range(dataset_size),
                    chunksize=4,
                ),
                total=dataset_size,
            )
        )

    data = torch.cat([torch.from_numpy(r[0]) for r in results])
    # each sample in results is turned into multiple frames, so we need to repeat the labels
    n_repeats: list[int] = [r[0].shape[0] for r in results]
    labels = torch.tensor([r[1] for r in results]).repeat_interleave(torch.tensor(n_repeats))
    return torch.utils.data.TensorDataset(data, labels)


def get_raw_datasets_with_split(prep_config: PrepareDataConfig) -> tuple[Any, Any, Any, tuple]:
    """Load raw datasets and apply train/test split WITHOUT any transforms

    This function is shared by both prepare.py and extract_metadata.py to ensure
    identical train/test splits.

    Args:
        prep_config: Preparation configuration

    Returns:
        Tuple of (dataset_train, dataset_test, sensor_size)
        Datasets contain raw events, no transforms applied
    """
    dataset_name = prep_config.name
    data_root = prep_config.data_root

    # Load raw datasets
    if dataset_name == "NMNIST":
        dataset_train_and_val = tonic.datasets.NMNIST(data_root, train=True)
        dataset_test = tonic.datasets.NMNIST(data_root, train=False)
        # Create a fixed validation split from training set if needed
        assert 0.0 < prep_config.val_split < 1.0, "val_split must be in (0.0, 1.0)"

        dataset_size = len(dataset_train_and_val)
        indices = np.arange(dataset_size)
        val_size = int(prep_config.val_split * dataset_size)
        train_size = dataset_size - val_size
        random_state = np.random.RandomState(prep_config.seed)
        random_state.shuffle(indices)

        dataset_train = SubsetDataset(dataset_train_and_val, indices[:train_size].tolist())
        dataset_val = SubsetDataset(dataset_train_and_val, indices[train_size:].tolist())

        logger.info(f"Train size: {len(dataset_train)}, Val size: {len(dataset_val)}, Test size: {len(dataset_test)}")

        sensor_size = tonic.datasets.NMNIST.sensor_size
    elif dataset_name == "CIFAR10DVS":
        dataset = tonic.datasets.CIFAR10DVS(data_root)
        # CIFAR10DVS does not have predefined train/test split so we need to split it ourselves now. the dataset is a list of tuple (events, label)
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)
        
        assert 0.0 < prep_config.val_split < 1.0, "val_split must be in (0.0, 1.0)"
        assert 0.0 < prep_config.test_split < 1.0, "test_split must be in (0.0, 1.0)"
        
        train_size = int((1 - prep_config.test_split) * dataset_size)
        val_size = int(prep_config.val_split * train_size)
        train_size = train_size - val_size
        random_state = np.random.RandomState(prep_config.seed)
        random_state.shuffle(indices)

        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:train_size + val_size].tolist()
        test_indices = indices[train_size + val_size:].tolist()

        dataset_train = SubsetDataset(dataset, train_indices)
        dataset_val = SubsetDataset(dataset, val_indices)
        dataset_test = SubsetDataset(dataset, test_indices)

        logger.info(f"Train size: {len(dataset_train)}, Val size: {len(dataset_val)}, Test size: {len(dataset_test)}")

        sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset_train, dataset_val, dataset_test, sensor_size


def create_transform(prep_config: PrepareDataConfig, sensor_size: tuple):
    """Create transform pipeline for converting events to frames

    Args:
        prep_config: Preparation configuration
        sensor_size: Sensor size tuple (H, W, C)

    Returns:
        Composed transform (Denoise + ToFrame + TransformPolarities)
    """
    # Configure ToFrame based on frame_mode
    if prep_config.frame_mode == "event_count":
        assert prep_config.events_per_frame is not None
        to_frame = tonic_transforms.ToFrame(
            sensor_size=sensor_size,
            event_count=prep_config.events_per_frame,
            overlap=int(prep_config.overlap * prep_config.events_per_frame),
        )
    else:  # time_window
        assert prep_config.time_window is not None
        to_frame = tonic_transforms.ToFrame(
            sensor_size=sensor_size,
            time_window=prep_config.time_window,
            overlap=int(prep_config.overlap * prep_config.time_window),
        )

    transforms = [
        to_frame,
        TransformPolarities(),
    ]
    if prep_config.denoise_time:
        transforms = [tonic_transforms.Denoise(filter_time=prep_config.denoise_time)] + transforms

    return tonic_transforms.Compose(transforms)  # type: ignore


def get_datasets(prep_config: PrepareDataConfig) -> tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """Load and preprocess raw dataset with tonic

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) wrapped with transforms
    """
    # Get raw datasets with correct split (shared logic)
    dataset_train, dataset_val, dataset_test, sensor_size = get_raw_datasets_with_split(prep_config)

    # Create transform pipeline
    transform = create_transform(prep_config, sensor_size)

    # Process datasets with transforms
    dataset_train = process_dataset(dataset_train, transform, prep_config)
    dataset_val = process_dataset(dataset_val, transform, prep_config)
    dataset_test = process_dataset(dataset_test, transform, prep_config)

    return dataset_train, dataset_val, dataset_test


def prepare_dataset(prep_config: PrepareDataConfig):
    """Prepare and cache dataset as TensorDatasets in NCHW format

    Saves to both scratch storage (high IO) and project storage (long-term)
    """
    dataset_name = prep_config.name
    logger.info(f"Preparing dataset: {dataset_name}")

    # Get processed datasets
    train_tensor_dataset, val_tensor_dataset, test_tensor_dataset = get_datasets(prep_config)

    # Create PreparedDataset with splits
    prepared_dataset = PreparedDataset(
        train=DatasetSplit("train", train_tensor_dataset),
        val=DatasetSplit("val", val_tensor_dataset),
        test=DatasetSplit("test", test_tensor_dataset)
    )

    # Log shapes
    logger.info(f"Training data shape: {prepared_dataset.train.data.shape}, labels shape: {prepared_dataset.train.labels.shape}")
    logger.info(f"Validation data shape: {prepared_dataset.val.data.shape}, labels shape: {prepared_dataset.val.labels.shape}")
    logger.info(f"Test data shape: {prepared_dataset.test.data.shape}, labels shape: {prepared_dataset.test.labels.shape}")

    # Save to both storages
    scratch_paths, project_paths = save_data_splits(
        dataset_name=dataset_name,
        prepared_dataset=prepared_dataset,
        cache_identifier=prep_config.get_cache_identifier()
    )

    # Log file sizes and locations
    logger.info("Data preparation complete.")
    logger.info(f"Cached data saved to:")
    logger.info(f"  Scratch storage (high IO):")
    for path in scratch_paths:
        logger.info(f"    {path} ({os.path.getsize(path) / (1024**2):.2f} MB)")
    logger.info(f"  Project storage (long-term):")
    for path in project_paths:
        logger.info(f"    {path} ({os.path.getsize(path) / (1024**2):.2f} MB)")