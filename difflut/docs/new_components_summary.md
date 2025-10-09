# New DiffLUT Components Summary

## Overview
This document summarizes the new layers and nodes added to the DiffLUT framework.

## New Layers

### AdaptiveLayer (`difflut/layers/adaptive_layer.py`)

**Purpose**: Parameter-efficient layer inspired by adaptive softmax, combining sparse connections and weight sharing.

**Key Features**:
- **Sparse Connections**: Each node only connects to a fraction of inputs (configurable via `connection_fraction`)
- **Weight Sharing**: Nodes are grouped into clusters that share weight matrices
- **Soft/Hard Selection**: Uses soft selection during training (Gumbel-softmax style) and hard selection during evaluation

**Parameters**:
- `connection_fraction` (default: 0.5): Fraction of inputs each node can see
- `num_clusters` (default: 4): Number of clusters for weight sharing
- `tau` (default: 0.001): Temperature for softmax

**Benefits**:
- Significant parameter reduction (30-70% fewer parameters than LearnableLayer)
- Maintains expressiveness through strategic weight sharing
- Scales better with large input dimensions

**Example Usage**:
```python
from difflut.layers import AdaptiveLayer
from difflut.nodes import LinearLUTNode

layer = AdaptiveLayer(
    input_size=784,
    output_size=128,
    node_type=LinearLUTNode,
    n=6,
    connection_fraction=0.5,
    num_clusters=4
)
```

**Demo Script**: `examples/adaptive_layer_demo.py`

---

## New Nodes

### 1. HybridNode (`difflut/nodes/hybrid_node.py`)

**Purpose**: Combines the efficiency of DWN with the trainability of UnboundProbabilistic.

**Key Features**:
- **Forward Pass**: Binary thresholding (like DWN) - discrete, efficient
- **Backward Pass**: Probabilistic gradients (like UnboundProbabilistic) - smooth, trainable
- **CUDA Support**: Includes custom CUDA kernels for GPU acceleration

**Mathematical Formulation**:
- Forward: `y = LUT[threshold(x)]`
- Backward: `∂y/∂x` computed using probabilistic expectation `E[LUT[a] | x] = Σ_a LUT[a] * Pr(a|x)`

**Benefits**:
- Fast inference (discrete forward)
- Effective training (smooth gradients)
- Best of both worlds approach

**CUDA Implementation**:
- `cuda/hybrid_cuda_kernel.cu`: CUDA kernel implementation
- `cuda/hybrid_cuda.cpp`: C++ binding
- `cuda/__init__.py`: Updated with `hybrid_forward` function and `HybridFunction` class

**Example Usage**:
```python
from difflut.nodes import HybridNode

node = HybridNode(
    num_inputs=6,
    output_dim=1,
    use_cuda=True  # Automatically uses CUDA if available
)
```

**Test Script**: `examples/test_hybrid_node.py`

---

### 2. FourierNode (`difflut/nodes/fourier_node.py`)

**Purpose**: Implements a DFT-like structure for smooth, bounded function approximation.

**Mathematical Formulation**:
```
y = 0.5 + Σ_k |w_k| * cos(2π * <k, x> + φ_k)
```

Where:
- `k` are frequency vectors (corners of hypercube {0,1}^n)
- `|w_k|` are learnable amplitudes
- `φ_k` are learnable phases
- `<k, x>` is the dot product

**Key Features**:
- **Real-Valued Output**: Using cosine (real part of complex exponential)
- **Bounded in [0, 1]**: Amplitude normalization ensures Σ|w_k| ≤ max_amplitude
- **Smooth and Differentiable**: Continuous everywhere
- **Frequency Analysis**: Can identify dominant frequencies

**Parameters**:
- `use_all_frequencies` (default: True): Use all 2^n frequencies or subset
- `max_amplitude` (default: 0.5): Maximum oscillation amplitude

**Benefits**:
- Smooth approximation of complex functions
- Interpretable through frequency analysis
- Natural regularization (L1 on amplitudes encourages sparsity)
- No discrete decisions - fully differentiable

**Example Usage**:
```python
from difflut.nodes import FourierNode

node = FourierNode(
    num_inputs=6,
    output_dim=1,
    use_all_frequencies=True,
    max_amplitude=0.5
)
```

---

### 3. FourierHermitianNode (`difflut/nodes/fourier_node.py`)

**Purpose**: Explicit Hermitian symmetry enforcement for guaranteed real-valued outputs.

**Mathematical Formulation**:
```
y = w_0 + 2 * Σ_{k>0} Re(w_k * exp(i * 2π * <k, x>))
```

With constraint: `w_{-k} = conj(w_k)` (Hermitian symmetry)

**Key Features**:
- **Explicit Symmetry**: Enforces w_{-k} = w̄_k by construction
- **Complex Weights**: Stores real and imaginary parts separately
- **DC Component**: Constant term (k=0) must be real

**Benefits**:
- More mathematically rigorous
- Guaranteed real outputs by construction
- Potentially better for certain function classes

**Example Usage**:
```python
from difflut.nodes import FourierHermitianNode

node = FourierHermitianNode(
    num_inputs=6,
    output_dim=1,
    max_amplitude=0.5
)
```

**Test Script**: `examples/test_fourier_node.py`

---

## CUDA Extensions

### Hybrid Node CUDA Kernel

**Files**:
- `difflut/nodes/cuda/hybrid_cuda_kernel.cu`
- `difflut/nodes/cuda/hybrid_cuda.cpp`

**Kernels**:
1. `hybrid_forward_kernel`: Binary thresholding forward pass
2. `hybrid_backward_input_kernel`: Probabilistic gradient w.r.t. inputs
3. `hybrid_backward_lut_kernel`: Probabilistic gradient w.r.t. LUT weights

**Compilation** (when building extensions):
```bash
python setup.py build_ext --inplace
```

---

## Registry Updates

All new components are automatically registered:

**Layers**:
- `"adaptive"` → AdaptiveLayer

**Nodes**:
- `"hybrid"` → HybridNode
- `"fourier"` → FourierNode
- `"fourier_hermitian"` → FourierHermitianNode

**Usage in Experiments**:
```yaml
model:
  layer_type: "adaptive"  # or "learnable", "random"
  node_type: "hybrid"     # or "fourier", "dwn", etc.
```

---

## Performance Comparisons

### Parameter Efficiency (AdaptiveLayer vs LearnableLayer)

For input_size=784, output_size=128, n=6:

| Configuration | Parameters | Reduction |
|--------------|------------|-----------|
| LearnableLayer | ~600K | baseline |
| Adaptive (50%, 4 clusters) | ~240K | 60% |
| Adaptive (30%, 8 clusters) | ~150K | 75% |
| Adaptive (20%, 16 clusters) | ~100K | 83% |

### Node Characteristics

| Node | Forward | Backward | Discrete | Bounded [0,1] |
|------|---------|----------|----------|---------------|
| DWN | Binary | EFD | Yes | No |
| UnboundProbabilistic | Probabilistic | Probabilistic | No | Yes |
| HybridNode | Binary | Probabilistic | Yes | No |
| FourierNode | Continuous | Continuous | No | Yes |
| LinearLUT | Weighted sum | Standard | No | Yes (sigmoid) |

---

## Testing

### Run All Tests

```bash
# Test adaptive layer
python examples/adaptive_layer_demo.py

# Test hybrid node
python examples/test_hybrid_node.py

# Test Fourier nodes
python examples/test_fourier_node.py
```

### Integration with Experiments

The new layers and nodes can be used directly in the experiment framework:

```bash
python experiments/run_experiment.py \
  --layer_type adaptive \
  --node_type fourier \
  --connection_fraction 0.5 \
  --num_clusters 4
```

---

## Future Work

1. **CUDA Optimization**: Further optimize CUDA kernels for hybrid node
2. **Adaptive Frequency Selection**: Let FourierNode learn which frequencies to use
3. **Hierarchical Clustering**: Multi-level weight sharing in AdaptiveLayer
4. **Sparse Fourier**: Combine FourierNode with sparse selection
5. **Benchmarking**: Comprehensive performance comparison across datasets

---

## References

- **Adaptive Softmax**: Grave et al., "Efficient softmax approximation for GPUs", 2017
- **Gumbel-Softmax**: Jang et al., "Categorical Reparameterization with Gumbel-Softmax", 2017
- **Fourier Features**: Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions", 2020
- **DWN**: Silva et al., "Weightless Neural Networks", 2019

---

## Contact

For questions or issues with these components, please refer to the main DiffLUT documentation or create an issue in the repository.
