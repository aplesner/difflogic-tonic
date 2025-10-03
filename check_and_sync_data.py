#!/usr/bin/env python3

import sys
import logging
from pathlib import Path

from src import io_funcs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def check_and_sync_dataset(dataset_name: str, files: dict[str, Path]):
    """Check and sync a single dataset between storage locations"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"{'='*60}")

    # Get file info for all locations using io_funcs
    scratch_train_info = io_funcs.get_file_info(files.get('scratch_train')) if 'scratch_train' in files else None
    scratch_test_info = io_funcs.get_file_info(files.get('scratch_test')) if 'scratch_test' in files else None
    project_train_info = io_funcs.get_file_info(files.get('project_train')) if 'project_train' in files else None
    project_test_info = io_funcs.get_file_info(files.get('project_test')) if 'project_test' in files else None

    needs_sync = False

    # Check train files
    if scratch_train_info or project_train_info:
        if scratch_train_info:
            size_mb, data_shape, labels_shape = scratch_train_info
        elif project_train_info:
            size_mb, data_shape, labels_shape = project_train_info
        else:
            size_mb, data_shape, labels_shape = (0.0, (), ())
        logger.info(f"Train: {files.get('scratch_train', 'N/A')} ({size_mb:.1f} MB)")
        logger.info(f"  Data: {data_shape}, Labels: {labels_shape}")

    # Check test files
    if scratch_test_info or project_test_info:
        if scratch_test_info:
            size_mb, data_shape, labels_shape = scratch_test_info
        elif project_test_info:
            size_mb, data_shape, labels_shape = project_test_info
        else:
            size_mb, data_shape, labels_shape = (0.0, (), ())
        logger.info(f"Test: {files.get('scratch_test', 'N/A')} ({size_mb:.1f} MB)")
        logger.info(f"  Data: {data_shape}, Labels: {labels_shape}")

    # Sync if needed using io_funcs
    if project_train_info or project_test_info:
        if not scratch_train_info or not scratch_test_info:
            logger.info("⚠ Files missing in scratch, copying from project...")
            needs_sync = True
            io_funcs.copy_cache_from_project_to_scratch(dataset_name)
        elif (project_train_info and scratch_train_info and project_train_info[0] != scratch_train_info[0]) or \
             (project_test_info and scratch_test_info and project_test_info[0] != scratch_test_info[0]):
            logger.warning(f"⚠ Size mismatch detected, re-syncing from project to scratch...")
            needs_sync = True
            io_funcs.copy_cache_from_project_to_scratch(dataset_name)
    elif not project_train_info and not project_test_info:
        logger.warning("⚠ Files missing in project storage")

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

    # Show storage directories using io_funcs
    scratch_dir = io_funcs.get_scratch_dir()
    project_dir = io_funcs.get_project_storage_dir()

    logger.info(f"\nStorage locations:")
    logger.info(f"  Scratch storage: {scratch_dir}")
    logger.info(f"  Project storage: {project_dir}")

    # Discover datasets using io_funcs
    datasets = io_funcs.discover_datasets()

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
