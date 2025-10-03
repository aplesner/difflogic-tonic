import os
from pathlib import Path
import shutil
import logging

import torch

from .config import Config

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


def get_data_paths(dataset_name: str, use_project_storage: bool = False) -> tuple[Path, Path]:
    """Get paths for tensor data

    Args:
        prep_config: Preparation configuration
        use_project_storage: If True, use project storage (long-term, IO bound)
                           If False, use scratch storage (short-term, high IO throughput)
    """
    if use_project_storage:
        # Use project storage paths (data_root/cache)
        storage_dir = get_project_storage_dir()
    else:
        # Use scratch storage for high IO throughput
        storage_dir = get_scratch_dir()
    storage_dir = storage_dir / 'data'
    storage_dir.mkdir(parents=True, exist_ok=True)

    train_path = storage_dir / f"{dataset_name}_train_data.pt"
    test_path = storage_dir / f"{dataset_name}_test_data.pt"

    return train_path, test_path


def save_data_splits(
        dataset_name: str, 
        train_tensor: torch.Tensor, 
        train_labels_tensor: torch.Tensor,
        test_tensor: torch.Tensor, 
        test_labels_tensor: torch.Tensor
    ) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    """Save dataset splits to both scratch and project storage"""
    # Get paths
    scratch_train_path, scratch_test_path = get_data_paths(dataset_name, use_project_storage=False)
    project_train_path, project_test_path = get_data_paths(dataset_name, use_project_storage=True)

    # Save to scratch storage
    torch.save({'data': train_tensor, 'labels': train_labels_tensor}, scratch_train_path)
    torch.save({'data': test_tensor, 'labels': test_labels_tensor}, scratch_test_path)

    # Save to project storage
    torch.save({'data': train_tensor, 'labels': train_labels_tensor}, project_train_path)
    torch.save({'data': test_tensor, 'labels': test_labels_tensor}, project_test_path)
    
    logger.info(f"Dataset cached:")
    logger.info(f"  Training: {train_tensor.shape}")
    logger.info(f"    Scratch: {scratch_train_path}")
    logger.info(f"    Project: {project_train_path}")
    logger.info(f"  Test: {test_tensor.shape}")
    logger.info(f"    Scratch: {scratch_test_path}")
    logger.info(f"    Project: {project_test_path}")

    return (scratch_train_path, scratch_test_path), (project_train_path, project_test_path)


def get_data_splits(dataset_name: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    # Get paths for both storage types
    scratch_train_path, scratch_test_path = get_data_paths(dataset_name, use_project_storage=False)
    project_train_path, project_test_path = get_data_paths(dataset_name, use_project_storage=True)

    scratch_exists = os.path.exists(scratch_train_path) and os.path.exists(scratch_test_path)
    project_exists = os.path.exists(project_train_path) and os.path.exists(project_test_path)

    # Determine cache location
    if scratch_exists:
        logger.info("Using cached data from scratch storage (fastest)")
        train_path, test_path = scratch_train_path, scratch_test_path
    elif project_exists:
        logger.info("Found cache in project storage, copying to scratch...")
        copy_cache_from_project_to_scratch(dataset_name)
        train_path, test_path = scratch_train_path, scratch_test_path
    else:
        # No cache found - raise error
        error_msg = (
            f"Cached data not found for dataset '{dataset_name}'\n"
            f"Checked locations:\n"
            f"  Scratch: {scratch_train_path} ({"missing" if not scratch_exists else "found"}) and {scratch_test_path} ({"missing" if not scratch_exists else "found"})\n"
            f"  Project: {project_train_path} ({"missing" if not project_exists else "found"}) and {project_test_path} ({"missing" if not project_exists else "found"})\n\n"
            f"Please run 'python3 prepare_data.py <config_file>' to prepare the dataset first."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    

    # Load cached TensorDatasets
    logger.info(f"Loading cached data from: {train_path}, {test_path}")
    train_cache = torch.load(train_path, map_location='cpu')
    test_cache = torch.load(test_path, map_location='cpu')

    return train_cache, test_cache

def copy_cache_from_project_to_scratch(dataset_name: str) -> bool:
    """Copy cached data from project storage to scratch storage if available

    Args:
        dataset_name: Name of the dataset

    Returns:
        True if successfully copied, False if project cache doesn't exist
    """
    project_train_path, project_test_path = get_data_paths(dataset_name, use_project_storage=True)
    scratch_train_path, scratch_test_path = get_data_paths(dataset_name, use_project_storage=False)

    if os.path.exists(project_train_path) and os.path.exists(project_test_path):
        logger.info(f"Copying cached data from project to scratch storage...")
        shutil.copy2(project_train_path, scratch_train_path)
        shutil.copy2(project_test_path, scratch_test_path)
        logger.info(f"  Copied: {project_train_path} -> {scratch_train_path}")
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

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
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

