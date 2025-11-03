# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy as np
cimport numpy as np
import cv2 as cv

def process_sample(batch, int count = 500, int overlap = 100, 
                   bint drop_incomplete = True) -> dict:
    cdef list x_list = []
    cdef list y_list = []
    cdef list t_list = []
    cdef list p_list = []
    cdef list label_list = []
    cdef list row_id_list = []
    cdef int i, start_idx, end_idx, length, n_windows, label
    
    for i in range(len(batch['x'])):
        x = batch['x'][i]
        y = batch['y'][i]
        t = batch['timestamp'][i]
        p = batch['polarity'][i]
        row_id = batch['row_id'][i]
        label = batch['label'][i]
        
        length = len(x)
        start_idx = 0
        n_windows = 0
        
        while start_idx + count <= length:
            end_idx = start_idx + count
            x_list.append(x[start_idx:end_idx])
            y_list.append(y[start_idx:end_idx])
            t_list.append(t[start_idx:end_idx])
            p_list.append(p[start_idx:end_idx])
            n_windows += 1
            start_idx += count - overlap
        
        if not drop_incomplete and start_idx < length:
            x_list.append(x[start_idx:])
            y_list.append(y[start_idx:])
            t_list.append(t[start_idx:])
            p_list.append(p[start_idx:])
            n_windows += 1
        
        label_list.extend([label] * n_windows)
        row_id_list.extend([row_id] * n_windows)
    
    return {
        'x': x_list, 
        'y': y_list, 
        'timestamp': t_list, 
        'polarity': p_list, 
        'label': label_list, 
        'row_id': row_id_list
    }

def zero_and_fill_canvas(np.ndarray[np.uint8_t, ndim=3] img, 
                np.ndarray[np.int32_t, ndim=1] x, 
                np.ndarray[np.int32_t, ndim=1] y, 
                np.ndarray[np.int32_t, ndim=1] p, 
                int padding) -> None:
    cdef int length = x.shape[0]
    cdef int xi, yi, pi
    cdef int j

    img.fill(0)
    
    for j in range(length):
        xi = x[j] + padding
        yi = y[j] + padding
        pi = p[j]
        img[yi, xi, pi] = 1


def event_stream_to_image(batch, tuple sensor_size, int padding = 1, 
                         bint denoise = False) -> dict:
    cdef np.ndarray[np.uint8_t, ndim=3] img
    cdef np.ndarray[np.uint8_t, ndim=2] kernel
    cdef int kernel_size = 2
    cdef int i, xi, yi, pi, length
    cdef list images = []
    cdef list timestamps = []
    cdef list durations = []
    cdef list labels = []
    cdef list n_events_list = []
    cdef list row_ids = []
    
    img = np.zeros(sensor_size, dtype=np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    assert isinstance(batch['x'][0], list), "Sample is not a list of lists"
    
    img = cv.copyMakeBorder(img, padding, padding, padding, padding, 
                           cv.BORDER_CONSTANT, value=0)
    
    for i in range(len(batch['x'])):
        x = batch['x'][i]
        y = batch['y'][i]
        p = batch['polarity'][i]
        t = batch['timestamp'][i]
        label = batch['label'][i]
        row_id = batch['row_id'][i]
        
        zero_and_fill_canvas(img, np.array(x, dtype=np.int32), 
                             np.array(y, dtype=np.int32), 
                             np.array(p, dtype=np.int32), padding)
        length = len(x)
        
        # Denoise
        if denoise:
            img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
            img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        
        # Extract final image
        if padding:
            _img = img[padding:-padding, padding:-padding, :].sum(axis=2) > 0
        else:
            _img = img.sum(axis=2) > 0
        
        images.append(_img)
        timestamps.append(t[0])
        durations.append(t[-1] - t[0])
        labels.append(label)
        n_events_list.append(length)
        row_ids.append(row_id)
    
    return {
        'image': images,
        'start_time': timestamps,
        'duration': durations,
        'label': labels,
        'n_events': n_events_list,
        'row_id': row_ids
    }
