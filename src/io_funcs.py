import os
from pathlib import Path
import shutil
import logging

import torch

from .config import Config
from .classes import PreparedDataset

logger = logging.getLogger(__name__)


def get_scratch_dir() -> Path:
    """Get storage directory, using SCRATCH_STORAGE_DIR if available, else 'storage/'"""
    scratch_dir = os.environ.get('SCRATCH_STORAGE_DIR')
    if scratch_dir:
        scratch_dir = Path(scratch_dir)
    else:
        scratch_dir = Path("./scratch/")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return scratch_dir

def get_project_storage_dir() -> Path:
    """Get project storage directory, using PROJECT_STORAGE_DIR if available, else 'data/'"""
    project_dir = os.environ.get('PROJECT_STORAGE_DIR')
    if project_dir:
        project_storage_dir = Path(project_dir)
    else:
        project_storage_dir = Path("./project_storage/")
    project_storage_dir.mkdir(parents=True, exist_ok=True)
    return project_storage_dir


def get_checkpoint_path(job_id: str) -> Path:
    """Get checkpoint path for a specific job"""
    storage_dir = get_scratch_dir() / 'checkpoints'
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir / f'checkpoint_{job_id}.pt'


def get_data_paths(dataset_name: str, use_project_storage: bool = False, cache_identifier: str | None = None) -> tuple[Path, Path, Path]:
    """Get paths for tensor data

    Args:
        dataset_name: Name of the dataset (e.g., "NMNIST", "CIFAR10DVS")
        use_project_storage: If True, use project storage (long-term, IO bound)
                           If False, use scratch storage (short-term, high IO throughput)
        cache_identifier: Optional identifier for cached data variant (e.g., "events_20000_overlap15000")

    Returns:
        Tuple of (train_path, val_path, test_path)
    """
    if use_project_storage:
        # Use project storage paths (data_root/cache)
        storage_dir = get_project_storage_dir()
    else:
        # Use scratch storage for high IO throughput
        storage_dir = get_scratch_dir()
    storage_dir = storage_dir / 'data'
    storage_dir.mkdir(parents=True, exist_ok=True)

    if cache_identifier:
        train_path = storage_dir / f"{dataset_name}_{cache_identifier}/train_data.pt"
        val_path = storage_dir / f"{dataset_name}_{cache_identifier}/val_data.pt"
        test_path = storage_dir / f"{dataset_name}_{cache_identifier}/test_data.pt"
    else:
        train_path = storage_dir / f"{dataset_name}/train_data.pt"
        val_path = storage_dir / f"{dataset_name}/val_data.pt"
        test_path = storage_dir / f"{dataset_name}/test_data.pt"

    # Ensure directories exist
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    return train_path, val_path, test_path


def save_data_splits(
        dataset_name: str,
        prepared_dataset: PreparedDataset,
        cache_identifier: str | None = None
    ) -> tuple[tuple[Path, Path, Path], tuple[Path, Path, Path]]:
    """Save dataset splits to both scratch and project storage

    Returns:
        Tuple of ((scratch_train, scratch_val, scratch_test), (project_train, project_val, project_test))
    """
    # Get paths
    ( scratch_train_path, scratch_val_path, scratch_test_path
    ) = get_data_paths(
        dataset_name, use_project_storage=False, cache_identifier=cache_identifier
    )
    
    (project_train_path, project_val_path, project_test_path
    ) = get_data_paths(
        dataset_name, use_project_storage=True, cache_identifier=cache_identifier
    )

    # Save to scratch storage
    torch.save({'data': prepared_dataset.train.data, 'labels': prepared_dataset.train.labels}, scratch_train_path)
    torch.save({'data': prepared_dataset.val.data, 'labels': prepared_dataset.val.labels}, scratch_val_path)
    torch.save({'data': prepared_dataset.test.data, 'labels': prepared_dataset.test.labels}, scratch_test_path)

    # Save to project storage
    torch.save({'data': prepared_dataset.train.data, 'labels': prepared_dataset.train.labels}, project_train_path)
    torch.save({'data': prepared_dataset.val.data, 'labels': prepared_dataset.val.labels}, project_val_path)
    torch.save({'data': prepared_dataset.test.data, 'labels': prepared_dataset.test.labels}, project_test_path)

    logger.info(f"Dataset cached:")
    logger.info(f"  Training: {prepared_dataset.train.data.shape}")
    logger.info(f"    Scratch: {scratch_train_path}")
    logger.info(f"    Project: {project_train_path}")
    logger.info(f"  Validation: {prepared_dataset.val.data.shape}")
    logger.info(f"    Scratch: {scratch_val_path}")
    logger.info(f"    Project: {project_val_path}")
    logger.info(f"  Test: {prepared_dataset.test.data.shape}")
    logger.info(f"    Scratch: {scratch_test_path}")
    logger.info(f"    Project: {project_test_path}")

    return (scratch_train_path, scratch_val_path, scratch_test_path), (project_train_path, project_val_path, project_test_path)


def get_data_splits(dataset_name: str, cache_identifier: str | None = None) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load dataset splits from cache

    Args:
        dataset_name: Name of the dataset (e.g., "CIFAR10DVS")
        cache_identifier: Optional cache variant identifier (e.g., "events_20000_overlap0_denoise5000")

    Returns:
        Tuple of (train_cache, val_cache, test_cache) dictionaries
    """
    # Get paths for both storage types
    scratch_train_path, scratch_val_path, scratch_test_path = get_data_paths(dataset_name, use_project_storage=False, cache_identifier=cache_identifier)
    project_train_path, project_val_path, project_test_path = get_data_paths(dataset_name, use_project_storage=True, cache_identifier=cache_identifier)

    scratch_exists = os.path.exists(scratch_train_path) and os.path.exists(scratch_val_path) and os.path.exists(scratch_test_path)
    project_exists = os.path.exists(project_train_path) and os.path.exists(project_val_path) and os.path.exists(project_test_path)

    # Determine cache location
    if scratch_exists:
        logger.info(f"Using cached data from scratch storage{' with identifier: ' + cache_identifier if cache_identifier else ''}")
        train_path, val_path, test_path = scratch_train_path, scratch_val_path, scratch_test_path
    elif project_exists:
        logger.info(f"Found cache in project storage{' with identifier: ' + cache_identifier if cache_identifier else ''}, copying to scratch...")
        copy_cache_from_project_to_scratch(dataset_name, cache_identifier)
        train_path, val_path, test_path = scratch_train_path, scratch_val_path, scratch_test_path
    else:
        # No cache found - raise error
        cache_msg = f" with identifier '{cache_identifier}'" if cache_identifier else ""
        error_msg = (
            f"Cached data not found for dataset '{dataset_name}'{cache_msg}\n"
            f"Checked locations:\n"
            f"  Scratch: {scratch_train_path} ({'missing' if not os.path.exists(scratch_train_path) else 'found'}), "
            f"{scratch_val_path} ({'missing' if not os.path.exists(scratch_val_path) else 'found'}), "
            f"{scratch_test_path} ({'missing' if not os.path.exists(scratch_test_path) else 'found'})\n"
            f"  Project: {project_train_path} ({'missing' if not os.path.exists(project_train_path) else 'found'}), "
            f"{project_val_path} ({'missing' if not os.path.exists(project_val_path) else 'found'}), "
            f"{project_test_path} ({'missing' if not os.path.exists(project_test_path) else 'found'})\n\n"
            f"Please run 'python3 prepare_data.py <config_file>' to prepare the dataset first."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


    # Load cached TensorDatasets
    logger.info(f"Loading cached data from: {train_path}, {val_path}, {test_path}")
    train_cache = torch.load(train_path, map_location='cpu', weights_only=True)
    val_cache = torch.load(val_path, map_location='cpu', weights_only=True)
    test_cache = torch.load(test_path, map_location='cpu', weights_only=True)

    return train_cache, val_cache, test_cache

def get_file_info(file_path: Path | None) -> tuple[float, tuple, tuple] | None:
    """Get file size and tensor shapes from a cached dataset file

    Args:
        file_path: Path to the .pt file

    Returns:
        Tuple of (size_mb, data_shape, labels_shape) or None if file doesn't exist
    """
    if not file_path or not file_path.exists():
        return None

    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        data = torch.load(file_path, map_location='cpu', weights_only=True)
        data_shape = tuple(data['data'].shape)
        labels_shape = tuple(data['labels'].shape)
        return size_mb, data_shape, labels_shape
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def discover_datasets() -> dict[str, dict[str, Path]]:
    """Discover all datasets in both scratch and project storage locations

    Returns:
        Dict mapping dataset names to their file paths:
        {
            'NMNIST': {
                'scratch_train': Path(...),
                'scratch_val': Path(...),
                'scratch_test': Path(...),
                'project_train': Path(...),
                'project_val': Path(...),
                'project_test': Path(...),
            }
        }
    """
    from collections import defaultdict

    scratch_dir = get_scratch_dir() / 'data'
    project_dir = get_project_storage_dir() / 'data'

    datasets = defaultdict(dict)

    # Scan scratch storage
    if scratch_dir.exists():
        for file_path in scratch_dir.glob('*_train_data.pt'):
            dataset_name = file_path.name.replace('_train_data.pt', '')
            datasets[dataset_name]['scratch_train'] = file_path

        for file_path in scratch_dir.glob('*_val_data.pt'):
            dataset_name = file_path.name.replace('_val_data.pt', '')
            datasets[dataset_name]['scratch_val'] = file_path

        for file_path in scratch_dir.glob('*_test_data.pt'):
            dataset_name = file_path.name.replace('_test_data.pt', '')
            datasets[dataset_name]['scratch_test'] = file_path

    # Scan project storage
    if project_dir.exists():
        for file_path in project_dir.glob('*_train_data.pt'):
            dataset_name = file_path.name.replace('_train_data.pt', '')
            datasets[dataset_name]['project_train'] = file_path

        for file_path in project_dir.glob('*_val_data.pt'):
            dataset_name = file_path.name.replace('_val_data.pt', '')
            datasets[dataset_name]['project_val'] = file_path

        for file_path in project_dir.glob('*_test_data.pt'):
            dataset_name = file_path.name.replace('_test_data.pt', '')
            datasets[dataset_name]['project_test'] = file_path

    return dict(datasets)


def copy_cache_from_project_to_scratch(dataset_name: str, cache_identifier: str | None = None) -> bool:
    """Copy cached data from project storage to scratch storage if available

    Args:
        dataset_name: Name of the dataset
        cache_identifier: Optional cache variant identifier

    Returns:
        True if successfully copied, False if project cache doesn't exist
    """
    project_train_path, project_val_path, project_test_path = get_data_paths(dataset_name, use_project_storage=True, cache_identifier=cache_identifier)
    scratch_train_path, scratch_val_path, scratch_test_path = get_data_paths(dataset_name, use_project_storage=False, cache_identifier=cache_identifier)

    if os.path.exists(project_train_path) and os.path.exists(project_val_path) and os.path.exists(project_test_path):
        logger.info(f"Copying cached data from project to scratch storage...")
        shutil.copy2(project_train_path, scratch_train_path)
        shutil.copy2(project_val_path, scratch_val_path)
        shutil.copy2(project_test_path, scratch_test_path)
        logger.info(f"  Copied: {project_train_path} -> {scratch_train_path}")
        logger.info(f"  Copied: {project_val_path} -> {scratch_val_path}")
        logger.info(f"  Copied: {project_test_path} -> {scratch_test_path}")
        return True
    return False


def save_checkpoint(model, optimizer, batch_count: int, config: Config, job_id: str, elapsed_time: float = 0.0):
    """Save training checkpoint"""
    checkpoint_path = get_checkpoint_path(job_id)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_count': batch_count,
        'elapsed_time': elapsed_time,
        'config_dict': config.model_dump()
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(job_id: str):
    """Load training checkpoint"""
    checkpoint_path = get_checkpoint_path(job_id)

    if not checkpoint_path.exists():
        logger.info(f"No checkpoint found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    return checkpoint


def copy_final_checkpoint(job_id: str, project_storage_path: str):
    """Copy final checkpoint from scratch to project storage"""
    src = get_checkpoint_path(job_id)
    dst = Path(project_storage_path) / f'final_checkpoint_{job_id}.pt'

    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info(f"Final checkpoint copied: {src} -> {dst}")
    else:
        logger.warning(f"Source checkpoint not found: {src}")


# Metadata I/O functions
def get_metadata_path(dataset_name: str, cache_identifier: str) -> Path:
    """Get path for metadata file

    Args:
        dataset_name: Name of dataset (e.g., "NMNIST", "CIFAR10DVS")
        cache_identifier: Cache identifier string (e.g., "events_20000_overlap0.75_denoise5000")

    Returns:
        Path to metadata JSON file
    """
    metadata_dir = get_project_storage_dir() / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir / f"{dataset_name}_{cache_identifier}_metadata.json"


def load_metadata(dataset_name: str, cache_identifier: str) -> dict | None:
    """Load metadata from JSON file

    Args:
        dataset_name: Name of dataset
        cache_identifier: Cache identifier string

    Returns:
        Metadata dictionary or None if file doesn't exist
    """
    import json

    metadata_path = get_metadata_path(dataset_name, cache_identifier)

    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    logger.info(f"Metadata loaded from: {metadata_path}")
    return metadata


def discover_metadata_files() -> dict[str, Path]:
    """Discover all metadata files in the metadata directory

    Returns:
        Dictionary mapping cache identifiers to file paths
    """
    metadata_dir = get_project_storage_dir() / 'metadata'

    if not metadata_dir.exists():
        return {}

    metadata_files = {}
    for file_path in metadata_dir.glob('*_metadata.json'):
        # Extract identifier from filename (remove dataset name and _metadata.json)
        # Format: DATASET_IDENTIFIER_metadata.json
        filename = file_path.stem.replace('_metadata', '')
        metadata_files[filename] = file_path

    return metadata_files

