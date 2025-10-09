"""Metadata extraction module for analyzing event data temporal characteristics

This module extracts temporal metadata from event datasets by simulating the
ToFrame transform without creating actual frame tensors. This allows analyzing:
- Sample durations
- Frame durations
- Event distribution across frames

The same train/test split and preprocessing parameters are used as in prepare.py.
"""

import os
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tonic
import tonic.transforms as tonic_transforms
import tqdm

from .config import PrepareDataConfig
from .prepare import get_raw_datasets_with_split
from ..io_funcs import get_project_storage_dir

logger = logging.getLogger(__name__)


class EventDurationExtractor:
    """Extract temporal metadata by simulating ToFrame without creating frames

    This class mimics the behavior of tonic.transforms.ToFrame to understand
    how events are split into frames, but only stores temporal metadata instead
    of actual frame tensors.
    """

    def __init__(self, prep_config: PrepareDataConfig, sensor_size: tuple):
        """
        Args:
            prep_config: Preparation configuration
            sensor_size: Sensor size tuple (H, W, C)
        """
        self.config = prep_config
        self.sensor_size = sensor_size

        # Create denoise transform if needed (same as prepare.py)
        if prep_config.denoise_time:
            self.denoise = tonic_transforms.Denoise(filter_time=prep_config.denoise_time)
        else:
            self.denoise = None

    def __call__(self, idx_and_sample):
        """Process one sample and extract duration metadata

        Args:
            idx_and_sample: Tuple of (index, (events, label))

        Returns:
            Dictionary with metadata for this sample
        """
        idx, (events, label) = idx_and_sample

        # Apply denoising if configured (same as prepare.py pipeline)
        if self.denoise:
            events = self.denoise(events)

        # Handle empty events case
        if len(events['t']) == 0:
            return {
                'sample_idx': idx,
                'label': int(label),
                'num_events_total': 0,
                'total_duration_us': 0,
                'frames': []
            }

        # Extract frame-level metadata
        if self.config.frame_mode == "event_count":
            frame_info = self._extract_event_count_durations(events)
        else:  # time_window
            frame_info = self._extract_time_window_durations(events)

        return {
            'sample_idx': idx,
            'label': int(label),
            'num_events_total': len(events['t']),
            'total_duration_us': int(events['t'][-1] - events['t'][0]),
            'frames': frame_info
        }

    def _extract_event_count_durations(self, events):
        """Simulate ToFrame with event_count mode

        Args:
            events: Structured array with 't', 'x', 'y', 'p' fields

        Returns:
            List of frame metadata dictionaries
        """
        assert self.config.events_per_frame is not None
        event_count = self.config.events_per_frame
        overlap = int(self.config.overlap * event_count)
        stride = event_count - overlap

        timestamps = events['t']
        num_events = len(timestamps)

        frame_info = []
        start_idx = 0

        # Iterate over complete frames
        while (start_idx + event_count) <= num_events:
            if start_idx + event_count < num_events:
                end_idx = start_idx + event_count
                frame_times = timestamps[start_idx:end_idx]
            else:
                frame_times = timestamps[start_idx:]

            if len(frame_times) > 0:
                duration_us = int(frame_times[-1] - frame_times[0])
                frame_info.append({
                    'duration_us': duration_us,
                    'num_events': len(frame_times),
                    'start_time_us': int(frame_times[0]),
                    'end_time_us': int(frame_times[-1])
                })

            # Move to next frame with stride
            start_idx += stride

        return frame_info

    def _extract_time_window_durations(self, events):
        """Simulate ToFrame with time_window mode

        Args:
            events: Structured array with 't', 'x', 'y', 'p' fields

        Returns:
            List of frame metadata dictionaries
        """
        assert self.config.time_window is not None
        time_window = self.config.time_window
        overlap_us = int(self.config.overlap * time_window)
        stride_us = time_window - overlap_us

        timestamps = events['t']

        if len(timestamps) == 0:
            return []

        start_time = timestamps[0]
        end_time = timestamps[-1]

        frame_info = []
        current_window_start = start_time

        while current_window_start <= end_time:
            window_end = current_window_start + time_window

            # Find events in this window
            mask = (timestamps >= current_window_start) & (timestamps < window_end)
            events_in_window = np.sum(mask)

            if events_in_window > 0:
                window_times = timestamps[mask]
                actual_duration = int(window_times[-1] - window_times[0])

                frame_info.append({
                    'duration_us': actual_duration,
                    'num_events': events_in_window,
                    'start_time_us': int(window_times[0]),
                    'end_time_us': int(window_times[-1]),
                    'window_start_us': int(current_window_start),
                    'window_end_us': int(window_end)
                })

            current_window_start += stride_us

        return frame_info


def extract_metadata_dataset(dataset, prep_config: PrepareDataConfig, sensor_size: tuple) -> list[dict]:
    """Extract metadata from dataset using multiprocessing

    Args:
        dataset: Tonic dataset (raw events, no transforms)
        prep_config: Preparation configuration
        sensor_size: Sensor size tuple

    Returns:
        List of metadata dictionaries, one per sample
    """
    extractor = EventDurationExtractor(prep_config, sensor_size)

    # Determine number of threads (same pattern as process_dataset)
    if prep_config.num_threads < 1:
        num_threads = os.cpu_count() or 1
        logger.info(f"Using all available CPU threads: {num_threads}")
    elif os.cpu_count() and prep_config.num_threads > os.cpu_count():  # type: ignore
        logger.warning(f"Requested {prep_config.num_threads} threads but only {os.cpu_count()} are available.")
        num_threads = os.cpu_count()
    else:
        num_threads = prep_config.num_threads

    logger.info(f"Extracting metadata with {num_threads} threads...")

    # Create indexed dataset for multiprocessing
    indexed_data = [(i, dataset[i]) for i in range(len(dataset))]

    with Pool(num_threads) as p:
        results = list(
            tqdm.tqdm(
                p.imap(extractor, indexed_data, chunksize=4),
                total=len(dataset),
                desc="Extracting metadata"
            )
        )

    return results


def compute_statistics(train_metadata: list[dict], test_metadata: list[dict]) -> dict:
    """Compute aggregate statistics from metadata

    Args:
        train_metadata: List of training sample metadata
        test_metadata: List of test sample metadata

    Returns:
        Dictionary with statistics for train and test splits
    """
    def compute_split_stats(metadata_list):
        """Compute statistics for one split"""
        if not metadata_list:
            return {}

        # Collect frame durations from all samples
        all_frame_durations = []
        all_frame_event_counts = []
        num_frames_per_event = []
        all_event_durations = []

        for sample in metadata_list:
            all_event_durations.append(sample['total_duration_us'])
            frames = sample['frames']
            num_frames_per_event.append(len(frames))

            for frame in frames:
                all_frame_durations.append(frame['duration_us'])
                all_frame_event_counts.append(frame['num_events'])

        stats: dict[str, float | int | dict[str, float | int]] = {
            'num_samples': len(metadata_list),
            'total_frames': len(all_frame_durations),
        }

        def get_stat_dict(values):
            return {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': int(np.min(values)),
                'max': int(np.max(values)),
            }

        if all_event_durations:
            stats["event duration (us)"] = get_stat_dict(all_event_durations)

        if all_frame_durations:
            stats["frame duration (us)"] = get_stat_dict(all_frame_durations)

        if all_frame_event_counts:
            stats["events per frame"] = get_stat_dict(all_frame_event_counts)

        if num_frames_per_event:
            stats["frames per event"] = get_stat_dict(num_frames_per_event)

        return stats

    return {
        'train': compute_split_stats(train_metadata),
        'test': compute_split_stats(test_metadata)
    }


def get_metadata_path(dataset_name: str, cache_identifier: str) -> Path:
    """Get path for metadata file

    Args:
        dataset_name: Name of dataset
        cache_identifier: Cache identifier string

    Returns:
        Path to metadata JSON file
    """
    metadata_dir = get_project_storage_dir() / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir / f"{dataset_name}_{cache_identifier}_metadata.json"


def save_metadata(
    dataset_name: str,
    cache_identifier: str,
    train_metadata: list[dict],
    test_metadata: list[dict],
    config: dict,
    statistics: dict
):
    """Save metadata to JSON file

    Args:
        dataset_name: Name of dataset
        cache_identifier: Cache identifier string
        train_metadata: List of training sample metadata
        test_metadata: List of test sample metadata
        config: Configuration dictionary
        statistics: Statistics dictionary
    """
    metadata_path = get_metadata_path(dataset_name, cache_identifier)

    metadata_output = {
        'dataset_name': dataset_name,
        'cache_identifier': cache_identifier,
        'config': config,
        'train_metadata': train_metadata,
        'test_metadata': test_metadata,
        'statistics': statistics
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata_output, f, indent=2)

    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"File size: {metadata_path.stat().st_size / (1024 ** 2):.2f} MB")


def extract_and_save_metadata(prep_config: PrepareDataConfig):
    """Main function: extract metadata for train/test splits and save to disk

    Args:
        prep_config: Preparation configuration
    """
    logger.info(f"Extracting metadata for dataset: {prep_config.name}")
    logger.info(f"Cache identifier: {prep_config.get_cache_identifier()}")

    # Get raw datasets (reuses prepare.py logic for identical split!)
    dataset_train, dataset_test, sensor_size = get_raw_datasets_with_split(prep_config)

    logger.info(f"Dataset sizes - Train: {len(dataset_train)}, Test: {len(dataset_test)}")

    # Extract metadata from both splits
    logger.info("Extracting training set metadata...")
    train_metadata = extract_metadata_dataset(dataset_train, prep_config, sensor_size)

    logger.info("Extracting test set metadata...")
    test_metadata = extract_metadata_dataset(dataset_test, prep_config, sensor_size)

    # Compute statistics
    logger.info("Computing statistics...")
    statistics = compute_statistics(train_metadata, test_metadata)

    # Log statistics
    logger.info("\n=== Statistics ===")
    logger.info(f"Training set:")
    for key, value in statistics['train'].items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for subkey, subvalue in value.items():
                logger.info(f"    {subkey}: {subvalue}")
        else:
            logger.info(f"    {key}: {value}")
    logger.info(f"Test set:")
    for key, value in statistics['test'].items():
        logger.info(f"  {key}:")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                logger.info(f"    {subkey}: {subvalue}")
        else:
            logger.info(f"    {key}: {value}")

    # Save metadata
    logger.info("\nSaving metadata...")
    save_metadata(
        dataset_name=prep_config.name,
        cache_identifier=prep_config.get_cache_identifier(),
        train_metadata=train_metadata,
        test_metadata=test_metadata,
        config=prep_config.model_dump(),
        statistics=statistics
    )

    logger.info("Metadata extraction complete!")
