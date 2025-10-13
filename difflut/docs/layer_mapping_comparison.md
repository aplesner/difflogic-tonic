# Layer Mapping Strategies Comparison

This document compares the different layer mapping strategies in the difflut library.

## Overview

All layers inherit from `BaseLUTLayer` and use LUT nodes, but differ in how they map inputs to node inputs.

## Layer Types

### 1. RandomLayer (`@register_layer("random")`)
- **Mapping**: Fixed random connections
- **Parameters**: 0 (mapping is fixed, not learned)
- **Use case**: Baseline, reproducible random connectivity
- **Advantages**: 
  - Zero mapping overhead
  - Ensures each input used at least once per node
  - Deterministic with seed
- **Configuration**:
  - `seed`: Random seed for reproducibility

---

### 2. LearnableLayer (`@register_layer("learnable")`)
- **Mapping**: Fully learnable connections via softmax
- **Parameters**: `output_size * n * input_size` weights in mapping module
- **Use case**: Maximum flexibility, learns optimal connections
- **Advantages**:
  - Can learn arbitrary optimal mappings
  - Soft selection during training, hard during inference
- **Disadvantages**:
  - High parameter count: O(input_size × output_size × n)
  - Memory intensive for large layers
- **Configuration**:
  - `tau`: Temperature for softmax (default: 0.001)

**Parameter count example**:
- input_size=784, output_size=100, n=6
- Mapping parameters: 784 × 100 × 6 = 470,400 parameters (just for mapping!)

---

### 3. CyclicLayer (`@register_layer("cyclic")`)
- **Mapping**: Sliding window over inputs
- **Parameters**: 0 (mapping is deterministic)
- **Use case**: Sequential/spatial data, parameter efficiency
- **Advantages**:
  - Zero mapping parameters
  - Natural for sequential or spatially organized data
  - Overlapping windows share information
- **Disadvantages**:
  - Fixed pattern, cannot adapt
- **Configuration**:
  - `stride`: Window step size (default: 1)
  - `wrap`: Whether to wrap around at end (default: True)

**Example** (input_size=10, output_size=4, n=3, stride=1):
```
Node 0: [0, 1, 2]
Node 1: [1, 2, 3]
Node 2: [2, 3, 4]
Node 3: [3, 4, 5]
```

---

### 4. StridedCyclicLayer (`@register_layer("strided_cyclic")`)
- **Mapping**: Sliding window with larger stride
- **Parameters**: 0 (mapping is deterministic)
- **Use case**: When you want less overlap between nodes
- **Advantages**:
  - Zero parameters
  - Non-overlapping windows (default stride=n)
  - Less redundancy between nodes
- **Configuration**:
  - `stride`: Window step size (default: n for non-overlapping)
  - `wrap`: Whether to wrap around at end (default: True)

**Example** (input_size=12, output_size=4, n=3, stride=3):
```
Node 0: [0, 1, 2]
Node 1: [3, 4, 5]
Node 2: [6, 7, 8]
Node 3: [9, 10, 11]
```

---

### 5. DilatedCyclicLayer (`@register_layer("dilated_cyclic")`)
- **Mapping**: Sliding window with dilation (skip connections)
- **Parameters**: 0 (mapping is deterministic)
- **Use case**: Increasing receptive field without parameters
- **Advantages**:
  - Zero parameters
  - Larger receptive field with same n
  - Similar to dilated convolutions
- **Configuration**:
  - `stride`: Window start position step (default: 1)
  - `dilation`: Spacing between elements in window (default: 2)
  - `wrap`: Whether to wrap around at end (default: True)

**Example** (input_size=10, output_size=4, n=3, stride=1, dilation=2):
```
Node 0: [0, 2, 4]
Node 1: [1, 3, 5]
Node 2: [2, 4, 6]
Node 3: [3, 5, 7]
```

---

## Parameter Comparison

For a typical layer with input_size=784, output_size=100, n=6:

| Layer Type | Mapping Parameters | LUT Parameters | Total |
|------------|-------------------|----------------|-------|
| RandomLayer | 0 | 100 × 2^6 | 6,400 |
| LearnableLayer | 470,400 | 100 × 2^6 | 476,800 |
| CyclicLayer | 0 | 100 × 2^6 | 6,400 |
| StridedCyclicLayer | 0 | 100 × 2^6 | 6,400 |
| DilatedCyclicLayer | 0 | 100 × 2^6 | 6,400 |

**Key insight**: LearnableLayer uses ~74× more parameters than cyclic variants!

---

## When to Use Each Layer

### Use **RandomLayer** when:
- You want a baseline with no inductive bias
- Memory is limited
- You need reproducible random connectivity

### Use **LearnableLayer** when:
- Input organization is unknown/arbitrary
- Maximum flexibility is needed
- Memory is not a constraint
- Data is small enough that parameter count is acceptable

### Use **CyclicLayer** when:
- Inputs have sequential/spatial structure
- You want parameter efficiency
- Overlapping receptive fields are beneficial
- Data has local correlations

### Use **StridedCyclicLayer** when:
- Inputs have sequential/spatial structure
- You want less redundancy between nodes
- Non-overlapping windows are preferred
- Memory is very limited

### Use **DilatedCyclicLayer** when:
- You need larger receptive fields
- Inputs have multi-scale structure
- You want to capture long-range dependencies
- Similar to using dilated convolutions

---

## Implementation Notes

All layers:
1. Inherit from `BaseLUTLayer`
2. Implement `get_mapping()` method
3. Use the parent's `forward()` method
4. Are registered with the `@register_layer` decorator
5. Can be inspected with `get_mapping_matrix()`

The cyclic variants (CyclicLayer, StridedCyclicLayer, DilatedCyclicLayer) store their mapping as a buffer, not a parameter, so it doesn't contribute to the model's parameter count or gradient computation.
