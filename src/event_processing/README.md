# Event Processing Cython Module

## Build

```bash
python setup.py build_ext --inplace
```

## Usage

```python
import event_processing

# Use the optimized functions
result = event_processing.process_sample(batch, count=500, overlap=100)
images = event_processing.event_stream_to_image(batch, sensor_size=(height, width, 2))
```

## Performance Notes

- Uses typed variables for faster execution
- Numpy memory views for array operations
- Disabled bounds checking for maximum speed
- Optimizations: -O3 and -march=native
