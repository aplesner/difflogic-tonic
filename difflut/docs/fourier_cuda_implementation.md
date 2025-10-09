# FourierNode CUDA Implementation

## Overview

This document describes the CUDA implementation for the FourierNode in the DiffLUT framework. The FourierNode implements a bounded discrete Fourier transform as a differentiable LUT operation.

## Architecture

The FourierNode computes:
```
y = 0.5 + Σ_k Re(w_k * exp(i * 2π * <k, x>))
```

Where:
- `k` are frequency vectors on the hypercube corners {0,1}^n
- `w_k` are complex weights represented as amplitude and phase: `w_k = |w_k| * exp(i * φ_k)`
- `x` is the input vector (mapped to [0, 1])
- The output is guaranteed to be real and bounded in [0, 1]

## File Structure

### CUDA Implementation Files

1. **fourier_cuda_kernel.cu**
   - Contains CUDA kernels for forward and backward passes
   - Implements three forward modes:
     - Training forward: Uses sigmoid on inputs for continuous gradients
     - Evaluation forward: Uses Heaviside step function for discrete behavior
   - Implements backward passes for all parameters

2. **fourier_cuda.cpp**
   - C++ bindings for PyTorch integration
   - Provides Python-accessible interface to CUDA kernels
   - Includes input validation and error checking

3. **cuda/__init__.py**
   - Python wrapper using PyTorch autograd
   - Provides `FourierFunction` autograd function
   - Includes CPU fallback implementation
   - Exports `fourier_forward()` function

4. **fourier_node.py**
   - Main FourierNode class
   - Integrates CUDA acceleration when available
   - Falls back to pure Python implementation on CPU

## CUDA Kernels

### Forward Pass (Training)

**Kernel**: `fourier_forward_kernel`

```cuda
__global__ void fourier_forward_kernel(
    const float* input,           // (batch_size, num_inputs)
    const float* frequencies,     // (num_frequencies, num_inputs)
    const float* amplitudes,      // (num_frequencies, output_dim)
    const float* phases,          // (num_frequencies, output_dim)
    const float* bias,            // (output_dim,)
    float* output,                // (batch_size, output_dim)
    ...
)
```

**Algorithm**:
1. Apply sigmoid to inputs: `x_sig = 1 / (1 + exp(-x))`
2. Compute dot products: `<k, x>` for all frequencies
3. Normalize amplitudes: `amp_norm = amp * max_amplitude / sum(amp)`
4. Compute Fourier sum: `Σ_k amp_norm_k * cos(2π * <k, x> + phase_k)`
5. Add bias and clamp to [0, 1]

**Thread organization**: Each thread handles one (batch, output_dim) pair

### Forward Pass (Evaluation)

**Kernel**: `fourier_forward_eval_kernel`

Similar to training forward, but uses Heaviside step function instead of sigmoid:
```cuda
x_heaviside[i] = (x[i] > 0.5) ? 1.0 : 0.0
```

This provides discrete, binary behavior during evaluation while maintaining the same Fourier computation structure.

### Backward Pass

The implementation includes four backward kernels:

#### 1. Input Gradients
**Kernel**: `fourier_backward_input_kernel`

Computes gradients w.r.t. inputs using chain rule:
```
∂L/∂x_i = Σ_{k,d} ∂L/∂y_d * ∂y_d/∂x_i
```

Where:
```
∂y_d/∂x_i = Σ_k (-amp_k * 2π * k_i * sin(2π<k,x> + phase_k)) * sigmoid'(x_i)
```

**Thread organization**: Each thread handles one (batch, input) pair

#### 2. Amplitude Gradients
**Kernel**: `fourier_backward_amplitude_kernel`

Computes gradients w.r.t. amplitudes:
```
∂L/∂amp_{k,d} = Σ_batch (amp_scale * cos(2π<k,x> + phase_k)) * ∂L/∂y_d
```

**Thread organization**: Each thread handles one (frequency, output_dim) pair

#### 3. Phase Gradients
**Kernel**: `fourier_backward_phase_kernel`

Computes gradients w.r.t. phases:
```
∂L/∂phase_{k,d} = Σ_batch (-amp_{k,d} * sin(2π<k,x> + phase_k)) * ∂L/∂y_d
```

**Thread organization**: Each thread handles one (frequency, output_dim) pair

#### 4. Bias Gradients
**Kernel**: `fourier_backward_bias_kernel`

Simple gradient computation:
```
∂L/∂bias_d = Σ_batch ∂L/∂y_d
```

**Thread organization**: Each thread handles one output dimension

## Usage

### Basic Usage

```python
from difflut.nodes.fourier_node import FourierNode

# Create node with CUDA acceleration
node = FourierNode(
    num_inputs=4,
    output_dim=2,
    use_all_frequencies=True,
    max_amplitude=0.5,
    use_cuda=True  # Enable CUDA
)

# Move to GPU
node = node.cuda()

# Forward pass (training)
x = torch.randn(32, 4).cuda()
output = node.forward_train(x)

# Forward pass (evaluation with Heaviside)
node.eval()
output_eval = node.forward_eval(x)

# Backward pass
loss = output.sum()
loss.backward()
```

### Compilation

To compile the CUDA extensions:

```bash
# Run the build script
bash scripts/build_cuda.sh

# Or manually
pip install -e . --no-build-isolation
```

### Testing

```bash
# Run the test script
python examples/test_fourier_cuda.py
```

## Performance Considerations

### Memory Layout
- All tensors are expected to be contiguous in memory
- Float32 precision is used throughout
- Frequency buffer is limited to 32 inputs maximum (can be adjusted)

### Optimization Strategies
1. **Coalesced Memory Access**: Threads access memory in patterns that allow for coalesced reads
2. **Fast Math**: Uses `--use_fast_math` compilation flag for faster trigonometric functions
3. **Shared Memory**: Could be added for frequency vectors in future optimization
4. **Thread Block Size**: Fixed at 256 threads per block (optimal for most GPUs)

### Scalability
- Forward pass: O(batch_size × output_dim × num_frequencies × num_inputs)
- Backward pass: Similar complexity
- Memory: O(batch_size × num_inputs + num_frequencies × output_dim)

## Differences from Python Implementation

1. **Input Processing**:
   - CUDA: Inline sigmoid/Heaviside computation in kernel
   - Python: Separate tensor operations

2. **Amplitude Normalization**:
   - CUDA: Per-output-dimension normalization in kernel
   - Python: Vectorized normalization

3. **Evaluation Mode**:
   - CUDA: Dedicated kernel with Heaviside function
   - Python: Applies Heaviside then calls standard forward

4. **Gradient Computation**:
   - CUDA: All gradients computed in parallel across batch
   - Python: Autograd handles gradient computation automatically

## Integration with Base Node

The FourierNode inherits from `BaseNode` and follows the standard interface:

- `forward_train(x)`: Training forward pass with continuous gradients
- `forward_eval(x)`: Evaluation forward pass with discrete (Heaviside) inputs
- `regularization_loss()`: L1 regularization on amplitudes
- `get_dominant_frequencies()`: Analysis method to identify important frequencies

## Future Enhancements

Possible optimizations and features:

1. **Shared Memory**: Cache frequency vectors in shared memory
2. **Half Precision**: Add FP16 support for faster training
3. **Kernel Fusion**: Combine normalization and cosine computation
4. **Dynamic Parallelism**: For very large numbers of frequencies
5. **Sparse Frequencies**: Only compute for non-zero frequency coefficients
6. **Batch Processing**: Optimize for very large batch sizes

## Troubleshooting

### CUDA Extension Not Found
```
ImportError: fourier_cuda not available
```
**Solution**: Run `bash scripts/build_cuda.sh` to compile extensions

### GPU Memory Issues
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or number of frequencies

### Incorrect Gradients
**Check**:
1. Inputs are contiguous: `x = x.contiguous()`
2. Correct dtype: `x = x.float()`
3. Requires grad: `x.requires_grad = True`

## References

1. DiffLUT Framework: Base architecture for differentiable LUT nodes
2. PyTorch CUDA Extensions: https://pytorch.org/tutorials/advanced/cpp_extension.html
3. Fourier Neural Operators: Inspiration for frequency-based neural networks
