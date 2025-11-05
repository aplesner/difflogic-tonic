import os
from dotenv import load_dotenv
from typing import Callable, Optional

import datasets
from datasets import load_dataset, ClassLabel
import numpy as np

from . import event_processing
from .data import SensorSizes

# source .env file to get HF_TOKEN
_ = load_dotenv()



def process_sample(batch: dict, count: int = 500, overlap: int | float = 100, 
                   drop_incomplete: bool = True) -> dict:
    if isinstance(overlap, float):
        assert 0 <= overlap < 1, "If overlap is a float, it must be in [0, 1)."
        overlap = int(count * overlap)
    x_list, y_list, t_list, p_list = [], [], [], []
    label_list, row_id_list = [], []
    for i in range(len(batch['x'])):
        x = batch['x'][i]
        y = batch['y'][i]
        t = batch['timestamp'][i]
        p = batch['polarity'][i]
        row_id = batch['row_id'][i]
        label = batch['label'][i]
        
        start_idx = 0
        
        while start_idx + count <= len(x):
            end_idx = start_idx + count
            x_list.append(x[start_idx:end_idx])
            y_list.append(y[start_idx:end_idx])
            t_list.append(t[start_idx:end_idx])
            p_list.append(p[start_idx:end_idx])
            start_idx += count - overlap
        
        if not drop_incomplete and start_idx < len(x):
            x_list.append(x[start_idx:])
            y_list.append(y[start_idx:])
            t_list.append(t[start_idx:])
            p_list.append(p[start_idx:])
        label_list.extend([label] * (len(x_list) - len(label_list)))
        row_id_list.extend([row_id] * (len(x_list) - len(row_id_list)))
    return {'x': x_list, 'y': y_list, 'timestamp': t_list, 'polarity': p_list, 'label': label_list, 'row_id': row_id_list}


def event_stream_to_image(batch: dict, sensor_size: tuple[int, int, int], denoise: Optional[Callable[[np.ndarray], np.ndarray]]) -> dict:
    img = np.zeros(sensor_size, dtype=np.uint8)

    images = []
    timestamps = []
    durations = []
    labels = []
    n_events_list = []
    row_ids = []


    for i in range(len(batch['x'])):
        x = batch['x'][i]
        y = batch['y'][i]
        p = batch['polarity'][i]
        t = batch['timestamp'][i]
        label = batch['label'][i]
        row_id = batch['row_id'][i]

        event_processing.zero_and_fill_canvas_list(img, x, y, p)

        _img = img.max(axis=2)  # convert to grayscale by averaging over polarity channels
        

        # Denoise the image with morphological operations (opening and closing)
        if denoise:
            _img = denoise(_img)

        _img = _img > 0  # binarize

        images.append(_img)
        timestamps.append(t[0])  # return the timestamp of the first event
        durations.append(t[-1] - t[0])  # return the duration of the event stream
        labels.append(label)
        n_events_list.append(len(x))
        row_ids.append(row_id)

    return {
        'image': images,
        'start_time': timestamps,
        'duration': durations,
        'label': labels,
        'n_events': n_events_list,
        'row_id': row_ids
    }



def load_and_process_data(dataset_name: str = 'aplesner-eth/NMNIST', split: str = 'test', num_samples: int = 50, num_classes: int = 10) -> datasets.Dataset:
    ds = load_dataset(dataset_name, split=split, token=os.getenv('HF_TOKEN'))
    # add a column with a row id
    ds = ds.map(lambda x, i: {'row_id': i}, with_indices=True)

    ds_small = ds.shuffle(seed=42).select(range(num_samples), keep_in_memory=True)  # type: ignore

    labels = ClassLabel(num_classes=num_classes)

    def convert_label(batch) -> dict:
        return {'label': labels.str2int(batch['label'])}
    ds_small = ds_small.map(convert_label, batched=True)  # type: ignore


    ds_new = ds_small.map(
        lambda batch: process_sample(batch, count=500, overlap=0.8, drop_incomplete=True),
        batched=True, remove_columns=ds_small.column_names, batch_size=5, desc="Pre-processing samples")

    if dataset_name.endswith('NMNIST'):
        sensor_size = SensorSizes.NMNIST.value
    else:
        raise ValueError(f"Unknown sensor size for dataset: {dataset_name}")

    ds_images = ds_new.map(
        lambda sample: event_stream_to_image(sample, sensor_size=sensor_size, denoise=True),
        batched=True, remove_columns=ds_new.column_names, batch_size=20, desc="Converting to images")

    return ds_images