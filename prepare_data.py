#!/usr/bin/env python3

import argparse
import sys
import logging
from pathlib import Path

from src.io_funcs import get_scratch_dir, get_project_storage_dir, get_data_paths
from src.prepare_data import PrepareDataConfig, prepare_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='Prepare and cache event-based dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 prepare_data.py config_nmnist.yaml
  python3 prepare_data.py config_cifar10dvs.yaml

This script processes raw event-based datasets and caches them as TensorDatasets
for fast loading during training. Data is saved to both scratch (high IO) and
project storage (long-term).

Environment Variables:
  SCRATCH_STORAGE_DIR - If set, uses this path for scratch storage and metadata
        '''
    )
    parser.add_argument('config_file', help='Config file path (e.g., config_nmnist.yaml)')

    args = parser.parse_args()

    if not Path(args.config_file).exists():
        logger.error(f"Config file '{args.config_file}' not found")
        sys.exit(1)

    logger.info(f"=== Data Preparation ===")
    logger.info(f"Config file: {args.config_file}")

    # Load preparation config from YAML
    prep_config = PrepareDataConfig.from_yaml(args.config_file)
    dataset_name = prep_config.name

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Events per frame: {prep_config.events_per_frame}")
    logger.info(f"Overlap: {prep_config.overlap}")
    logger.info(f"Denoise time: {prep_config.denoise_time}")
    logger.info(f"Reset cache: {prep_config.reset_cache}")
    logger.info("")

    # Show storage directories
    scratch_dir = get_scratch_dir()
    project_dir = get_project_storage_dir()

    logger.info(f"Storage locations:")
    logger.info(f"  Scratch storage: {scratch_dir}")
    logger.info(f"  Project storage: {project_dir}")
    logger.info("")


    # Check if cache exists and reset_cache is False
    scratch_train_path, scratch_test_path = get_data_paths(dataset_name, use_project_storage=False)
    project_train_path, project_test_path = get_data_paths(dataset_name, use_project_storage=True)

    scratch_exists = Path(scratch_train_path).exists() and Path(scratch_test_path).exists()
    project_exists = Path(project_train_path).exists() and Path(project_test_path).exists()

    if not prep_config.reset_cache and (scratch_exists or project_exists):
        logger.info("Cache already exists and reset_cache=False. Skipping preparation.")
        if scratch_exists:
            logger.info(f"  Using cache from scratch storage:")
            logger.info(f"    Train: {scratch_train_path}")
            logger.info(f"    Test: {scratch_test_path}")
        elif project_exists:
            logger.info(f"  Using cache from project storage:")
            logger.info(f"    Train: {project_train_path}")
            logger.info(f"    Test: {project_test_path}")
        logger.info("")
    else:
        # Prepare dataset
        logger.info("Preparing dataset (this may take a while)...")
        prepare_dataset(prep_config)
        logger.info("")

    # Show final cache status (paths already computed above)
    logger.info("=== Cache Status ===")

    if Path(scratch_train_path).exists():
        size_mb = Path(scratch_train_path).stat().st_size / (1024*1024)
        logger.info(f"✓ Scratch train: {scratch_train_path} ({size_mb:.1f} MB)")
    else:
        logger.warning(f"✗ Scratch train: {scratch_train_path} (missing)")

    if Path(scratch_test_path).exists():
        size_mb = Path(scratch_test_path).stat().st_size / (1024*1024)
        logger.info(f"✓ Scratch test: {scratch_test_path} ({size_mb:.1f} MB)")
    else:
        logger.warning(f"✗ Scratch test: {scratch_test_path} (missing)")

    if Path(project_train_path).exists():
        size_mb = Path(project_train_path).stat().st_size / (1024*1024)
        logger.info(f"✓ Project train: {project_train_path} ({size_mb:.1f} MB)")
    else:
        logger.warning(f"✗ Project train: {project_train_path} (missing)")

    if Path(project_test_path).exists():
        size_mb = Path(project_test_path).stat().st_size / (1024*1024)
        logger.info(f"✓ Project test: {project_test_path} ({size_mb:.1f} MB)")
    else:
        logger.warning(f"✗ Project test: {project_test_path} (missing)")

    logger.info("")
    logger.info("Data preparation complete! You can now run training.")

if __name__ == "__main__":
    main()