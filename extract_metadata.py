#!/usr/bin/env python3
"""Extract temporal metadata from event datasets

This script processes event data to extract sample durations by simulating
the ToFrame transform without storing actual frames.

Uses the same configuration files and train/test split as prepare_data.py
to ensure consistency.

Usage:
    python3 extract_metadata.py configs/prepare_data/cifar10dvs_events_20k.yaml
    python3 extract_metadata.py configs/prepare_data/nmnist.yaml

Output:
    metadata/DATASET_CACHE_IDENTIFIER_metadata.json
"""

import argparse
import sys
import logging
from pathlib import Path

from src.prepare_data import PrepareDataConfig
from src.prepare_data.extract_metadata import extract_and_save_metadata
from src.io_funcs import get_project_storage_dir

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Extract temporal metadata from event datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 extract_metadata.py configs/prepare_data/nmnist.yaml
  python3 extract_metadata.py configs/prepare_data/cifar10dvs_events_20k.yaml
  python3 extract_metadata.py configs/prepare_data/cifar10dvs_time_50ms.yaml

This script analyzes event datasets to extract temporal characteristics:
- Frame durations (actual time covered by events in each frame)
- Event counts per frame
- Total sample durations
- Distribution statistics

The same preprocessing parameters and train/test split are used as in
prepare_data.py to ensure consistency.

Output is saved to metadata/ directory as JSON files.
        '''
    )
    parser.add_argument('config_file', help='Config file from configs/prepare_data/')

    args = parser.parse_args()

    if not Path(args.config_file).exists():
        logger.error(f"Config file '{args.config_file}' not found")
        sys.exit(1)

    logger.info(f"=== Metadata Extraction ===")
    logger.info(f"Config file: {args.config_file}")

    # Load preparation config from YAML (same as prepare_data.py)
    prep_config = PrepareDataConfig.from_yaml(args.config_file)
    dataset_name = prep_config.name

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Frame mode: {prep_config.frame_mode}")
    if prep_config.frame_mode == "event_count":
        logger.info(f"Events per frame: {prep_config.events_per_frame}")
    else:
        logger.info(f"Time window: {prep_config.time_window} μs")
    logger.info(f"Overlap: {prep_config.overlap}")
    logger.info(f"Denoise time: {prep_config.denoise_time}")
    logger.info(f"Cache identifier: {prep_config.get_cache_identifier()}")
    logger.info("")

    # Show output directory
    metadata_dir = get_project_storage_dir() / 'metadata'
    logger.info(f"Output directory: {metadata_dir}")
    logger.info("")

    # Extract metadata
    logger.info("Starting metadata extraction (this may take a while)...")
    logger.info("")

    try:
        extract_and_save_metadata(prep_config)
    except Exception as e:
        logger.error(f"Error during metadata extraction: {e}", exc_info=True)
        sys.exit(1)

    logger.info("")
    logger.info("=== Metadata Extraction Complete ===")


if __name__ == "__main__":
    main()
