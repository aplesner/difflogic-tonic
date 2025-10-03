import os
import logging
from multiprocessing import Pool
from pathlib import Path


import numpy as np
import torch
import tonic
import tonic.transforms as tonic_transforms
import tqdm

from .config import PrepareDataConfig
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

    with Pool(prep_config.num_threads) as p:
        dataset_size = len(dataset)
        # dataset_size = min(len(dataset), 10)
        results = list(
            tqdm.tqdm(
                p.imap(
                    processor,
                    range(dataset_size),
                    chunksize=2,
                ),
                total=dataset_size,
            )
        )
    data = torch.cat([torch.from_numpy(r[0]) for r in results])
    # each sample in results is turned into multiple frames, so we need to repeat the labels
    n_repeats: list[int] = [r[0].shape[0] for r in results]
    labels = torch.tensor([r[1] for r in results]).repeat_interleave(torch.tensor(n_repeats))
    return torch.utils.data.TensorDataset(data, labels)


def get_dataset(prep_config: PrepareDataConfig) -> tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """Load and preprocess raw dataset with tonic

    Returns:
        Tuple of (train_dataset, test_dataset) wrapped with transforms
    """
    dataset_name = prep_config.name
    data_root = prep_config.data_root

    # Load raw datasets
    if dataset_name == "NMNIST":
        dataset_train = tonic.datasets.NMNIST(data_root, train=True)
        dataset_test = tonic.datasets.NMNIST(data_root, train=False)
        dataset = None
        sensor_size = tonic.datasets.NMNIST.sensor_size
    elif dataset_name == "CIFAR10DVS":
        dataset = tonic.datasets.CIFAR10DVS(data_root)
        dataset_train, dataset_test = None, None
        sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Setup transforms
    transforms = [
        tonic_transforms.ToFrame(
            sensor_size=sensor_size,
            event_count=prep_config.events_per_frame,
            overlap=prep_config.overlap,
        ),
        TransformPolarities(),
    ]
    if prep_config.denoise_time:
        transforms = [tonic_transforms.Denoise(filter_time=prep_config.denoise_time)] + transforms

    transform = tonic_transforms.Compose(transforms)  # type: ignore

    
    if dataset is not None:
        # Process the dataset and split into train/tests
        dataset = process_dataset(dataset, transform, prep_config)
        n = len(dataset)
        split = int(prep_config.test_split * n)
        dataset_train, dataset_test = torch.utils.data.random_split(
            dataset,
            [n - split, split],
            generator=torch.Generator().manual_seed(prep_config.seed)
        )
        # Create TensorDatasets from the subsets
        dataset_train = torch.utils.data.TensorDataset(
            torch.stack([dataset_train[i][0] for i in range(len(dataset_train))]),
            torch.tensor([dataset_train[i][1] for i in range(len(dataset_train))])
        )
        dataset_test = torch.utils.data.TensorDataset(
            torch.stack([dataset_test[i][0] for i in range(len(dataset_test))]),
            torch.tensor([dataset_test[i][1] for i in range(len(dataset_test))])
        )

    else:
        # Datasets are already split. Process them separately
        dataset_train = process_dataset(dataset_train, transform, prep_config)
        dataset_test = process_dataset(dataset_test, transform, prep_config)

    return dataset_train, dataset_test


def prepare_dataset(prep_config: PrepareDataConfig):
    """Prepare and cache dataset as TensorDatasets in NCHW format

    Saves to both scratch storage (high IO) and project storage (long-term)
    """
    dataset_name = prep_config.name
    logger.info(f"Preparing dataset: {dataset_name}")

    # Get original datasets
    dataset_train, dataset_test = get_dataset(prep_config)

    # Convert to tensors
    train_tensor, train_labels_tensor = dataset_train.tensors
    test_tensor, test_labels_tensor = dataset_test.tensors

    # Log the shapes
    logger.info(f"Training data shape: {train_tensor.shape}, labels shape: {train_labels_tensor.shape}")
    logger.info(f"Test data shape: {test_tensor.shape}, labels shape: {test_labels_tensor.shape}")

    # Save to both storages
    (scratch_train_path, scratch_test_path), (project_train_path, project_test_path) = save_data_splits(
        dataset_name=dataset_name,
        train_tensor=train_tensor,
        train_labels_tensor=train_labels_tensor,
        test_tensor=test_tensor,
        test_labels_tensor=test_labels_tensor
    )

    logger.info("Data preparation complete.")
    # log file sizes and locations
    logger.info(f"Cached data saved to:")
    logger.info(f"  Scratch storage (high IO):")
    logger.info(f"    {scratch_train_path} ({os.path.getsize(scratch_train_path) / (1024**2):.2f} MB)")
    logger.info(f"    {scratch_test_path}   ({os.path.getsize(scratch_test_path) / (1024**2):.2f} MB)")
    logger.info(f"  Project storage (long-term):")
    logger.info(f"    {project_train_path} ({os.path.getsize(project_train_path) / (1024**2):.2f} MB)")
    logger.info(f"    {project_test_path}   ({os.path.getsize(project_test_path) / (1024**2):.2f} MB)")