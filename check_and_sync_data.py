#!/usr/bin/env python3

import os
import sys
import logging
import shutil
from pathlib import Path
from collections import defaultdict

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def get_storage_dir() -> Path:
    """Get storage directory for cached datasets

    Uses SCRATCH_STORAGE_DIR if available, else 'storage/'
    """
    scratch_dir = os.environ.get('SCRATCH_STORAGE_DIR')
    if scratch_dir:
        return Path(scratch_dir) / 'storage'
    return Path('storage')


def get_project_storage_dir() -> Path:
    """Get project storage directory for cached datasets

    Uses SCRATCH_STORAGE_DIR/data/cache if available, else 'data/cache'
    """
    scratch_dir = os.environ.get('SCRATCH_STORAGE_DIR')
    if scratch_dir:
        return Path(scratch_dir) / 'data' / 'cache'
    return Path('data') / 'cache'


def discover_datasets() -> dict[str, dict[str, Path]]:
    """Discover all datasets in both storage locations

    Returns:
        Dict mapping dataset names to their file paths:
        {
            'NMNIST': {
                'scratch_train': Path(...),
                'scratch_test': Path(...),
                'project_train': Path(...),
                'project_test': Path(...),
            }
        }
    """
    scratch_dir = get_storage_dir()
    project_dir = get_project_storage_dir()

    datasets = defaultdict(dict)

    # Scan scratch storage
    if scratch_dir.exists():
        for file_path in scratch_dir.glob('*_train_data.pt'):
            dataset_name = file_path.name.replace('_train_data.pt', '')
            datasets[dataset_name]['scratch_train'] = file_path

        for file_path in scratch_dir.glob('*_test_data.pt'):
            dataset_name = file_path.name.replace('_test_data.pt', '')
            datasets[dataset_name]['scratch_test'] = file_path

    # Scan project storage
    if project_dir.exists():
        for file_path in project_dir.glob('*_train_data.pt'):
            dataset_name = file_path.name.replace('_train_data.pt', '')
            datasets[dataset_name]['project_train'] = file_path

        for file_path in project_dir.glob('*_test_data.pt'):
            dataset_name = file_path.name.replace('_test_data.pt', '')
            datasets[dataset_name]['project_test'] = file_path

    return dict(datasets)


def get_file_info(file_path: Path | None) -> tuple[float, tuple, tuple] | None:
    """Get file size and tensor shapes from a cached dataset file

    Returns:
        Tuple of (size_mb, data_shape, labels_shape) or None if file doesn't exist
    """
    if not file_path or not file_path.exists():
        return None

    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        data = torch.load(file_path)
        data_shape = tuple(data['data'].shape)
        labels_shape = tuple(data['labels'].shape)
        return size_mb, data_shape, labels_shape
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def sync_to_scratch(project_path: Path, scratch_path: Path) -> bool:
    """Copy file from project storage to scratch storage

    Returns:
        True if copy was successful, False otherwise
    """
    try:
        scratch_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_path, scratch_path)
        logger.info(f"  Copied: {project_path} -> {scratch_path}")
        return True
    except Exception as e:
        logger.error(f"  Failed to copy {project_path} to {scratch_path}: {e}")
        return False


def check_and_sync_dataset(dataset_name: str, files: dict[str, Path]):
    """Check and sync a single dataset between storage locations"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"{'='*60}")

    # Get file info for all locations
    scratch_train_info = get_file_info(files.get('scratch_train')) if 'scratch_train' in files else None
    scratch_test_info = get_file_info(files.get('scratch_test')) if 'scratch_test' in files else None
    project_train_info = get_file_info(files.get('project_train')) if 'project_train' in files else None
    project_test_info = get_file_info(files.get('project_test')) if 'project_test' in files else None

    needs_sync = False

    # Check train files
    if scratch_train_info or project_train_info:
        if scratch_train_info:
            size_mb, data_shape, labels_shape = scratch_train_info
        elif project_train_info:
            size_mb, data_shape, labels_shape = project_train_info
        else:
            size_mb, data_shape, labels_shape = (0.0, (), ())
        logger.info(f"Train: {files['scratch_train']} ({size_mb:.1f} MB)")
        logger.info(f"  Data: {data_shape}, Labels: {labels_shape}")

    # Sync train if needed
    if project_train_info and not scratch_train_info:
        logger.info("⚠ Train file missing in scratch, copying from project...")
        needs_sync = True
        if sync_to_scratch(files['project_train'], files.get('scratch_train') or get_storage_dir() / files['project_train'].name):
            scratch_train_info = get_file_info(files.get('scratch_train') or get_storage_dir() / files['project_train'].name)
    elif project_train_info and scratch_train_info:
        if project_train_info[0] != scratch_train_info[0]:
            logger.warning(f"⚠ Size mismatch! Project: {project_train_info[0]:.1f} MB, Scratch: {scratch_train_info[0]:.1f} MB")
            logger.info("  Copying from project to scratch...")
            needs_sync = True
            sync_to_scratch(files['project_train'], files['scratch_train'])
    elif not project_train_info:
        logger.warning("⚠ Train file missing in project storage")

    # Check test files
    if scratch_test_info or project_test_info:
        if scratch_test_info:
            size_mb, data_shape, labels_shape = scratch_test_info
        elif project_test_info:
            size_mb, data_shape, labels_shape = project_test_info
        else:
            size_mb, data_shape, labels_shape = (0.0, (), ())
        logger.info(f"Test: {files['scratch_test']} ({size_mb:.1f} MB)")
        logger.info(f"  Data: {data_shape}, Labels: {labels_shape}")

    # Sync test if needed
    if project_test_info and not scratch_test_info:
        logger.info("⚠ Test file missing in scratch, copying from project...")
        needs_sync = True
        if sync_to_scratch(files['project_test'], files.get('scratch_test') or get_storage_dir() / files['project_test'].name):
            scratch_test_info = get_file_info(files.get('scratch_test') or get_storage_dir() / files['project_test'].name)
    elif project_test_info and scratch_test_info:
        if project_test_info[0] != scratch_test_info[0]:
            logger.warning(f"⚠ Size mismatch! Project: {project_test_info[0]:.1f} MB, Scratch: {scratch_test_info[0]:.1f} MB")
            logger.info("  Copying from project to scratch...")
            needs_sync = True
            sync_to_scratch(files['project_test'], files['scratch_test'])

    # Status
    logger.info("\n--- Status ---")
    if scratch_train_info and scratch_test_info and project_train_info and project_test_info:
        if not needs_sync:
            logger.info("✓ Synced")
        else:
            logger.info("✓ Now synced (after copying)")
    elif scratch_train_info and scratch_test_info and not project_train_info and not project_test_info:
        logger.warning("⚠ Only in scratch storage")
    elif project_train_info and project_test_info and not scratch_train_info and not scratch_test_info:
        logger.warning("⚠ Only in project storage (needs sync)")
    else:
        logger.error("✗ Incomplete dataset (missing files)")


def main():
    logger.info("=== Dataset Cache Check and Sync ===")

    # Show storage directories
    scratch_dir = get_storage_dir()
    project_dir = get_project_storage_dir()

    logger.info(f"\nStorage locations:")
    logger.info(f"  Scratch storage: {scratch_dir}")
    logger.info(f"  Project storage: {project_dir}")

    # Discover datasets
    datasets = discover_datasets()

    if not datasets:
        logger.warning("\nNo datasets found in either storage location!")
        logger.info("Run prepare_data.py to create cached datasets.")
        sys.exit(0)

    logger.info(f"\nFound {len(datasets)} dataset(s): {', '.join(datasets.keys())}")

    # Check and sync each dataset
    for dataset_name, files in sorted(datasets.items()):
        check_and_sync_dataset(dataset_name, files)

    logger.info(f"\n{'='*60}")
    logger.info("Check and sync complete!")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
