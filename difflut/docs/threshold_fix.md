# DWN and Hybrid Node Threshold Fix

## Problem

The DWN and Hybrid nodes were performing poorly (10-15% accuracy on MNIST) while the Probabilistic node achieved 85%+ accuracy. The root cause was a **threshold mismatch** between the encoder output range and the node's binary conversion.

### Root Cause

1. **Thermometer encoder** outputs continuous values in the range `[0, 1]`
2. **DWN and Hybrid nodes** were thresholding at `0.0` to convert to binary:
   ```python
   x_binary = (x > 0).float()  # Wrong for [0, 1] inputs!
   ```
3. Since thermometer values are always `≥ 0`, everything became `1`, **losing all information**
4. All inputs collapsed to the same LUT address, making the network unable to learn

### Why Probabilistic Node Worked

The Probabilistic node is designed for continuous `[0, 1]` inputs and uses them directly in probabilistic computations:
```python
# Computes Pr(a|x) = ∏_j [x_j^a_j * (1-x_j)^(1-a_j)]
```
It doesn't threshold, so it preserves the input information.

## Solution

Changed the threshold from `0.0` to `0.5` to properly handle `[0, 1]` range inputs:

```python
# Before (broken)
x_binary = (x > 0).float()

# After (fixed)
x_binary = (x > 0.5).float()
```

This change was applied to:

### 1. DWN Node - Python Implementation
**File:** `difflut/nodes/dwn_node.py`
**Line:** ~103

```python
def _forward_python(self, x: torch.Tensor) -> torch.Tensor:
    # Binary threshold at 0.5 for [0, 1] inputs
    x_binary = (x > 0.5).float()
```

### 2. DWN Node - CUDA Kernel
**File:** `difflut/nodes/cuda/efd_cuda_kernel.cu`
**Lines:** 20, 76

```cuda
// Forward kernel
uint addr = input[i][mapping[j][0]] > 0.5f;
for(int l = 1; l < mapping.size(1); ++l)
    addr |= (uint)(input[i][mapping[j][l]] > 0.5f) << l;

// Backward kernel
uint addr = input[i][mapping[j][0]] > 0.5f;
for(int l = 1; l < mapping.size(1); ++l) {
    addr |= (uint)(input[i][mapping[j][l]] > 0.5f) << l;
}
```

### 3. Hybrid Node - Python Implementation
**File:** `difflut/nodes/hybrid_node.py`
**Line:** ~167

```python
@staticmethod
def forward(ctx, x, luts, binary_combinations, num_inputs, output_dim):
    # Forward: Binary thresholding at 0.5 for [0, 1] inputs
    x_binary = (x > 0.5).float()
```

### 4. Hybrid Node - CUDA Kernel
**File:** `difflut/nodes/cuda/hybrid_cuda_kernel.cu`
**Line:** ~31

```cuda
// Compute LUT address using binary thresholding
int addr = 0;
for (int i = 0; i < n; i++) {
    int input_idx = mapping[lut_idx * n + i];
    float val = input[batch_idx * input_length + input_idx];
    if (val > 0.5f) {
        addr |= (1 << i);
    }
}
```

## Testing

After applying the fix, you need to:

1. **Recompile CUDA extensions** (since we modified `.cu` files):
   ```bash
   cd /itet-stor/sbuehrer/net_scratch/difflut
   rm -f *.so  # Remove old compiled extensions
   pip install -e . --force-reinstall --no-deps
   ```

2. **Run test script** to verify the fix:
   ```bash
   python tests/test_threshold_fix.py
   ```

3. **Re-run experiments** with DWN and Hybrid nodes:
   ```bash
   # The experiments should now show much better performance
   python experiments/run_experiment.py --config experiments/configs/your_config.yaml
   ```

## Expected Results

With the fix applied:
- **DWN node** should now achieve similar performance to Probabilistic node (~80-90% on MNIST)
- **Hybrid node** should also show significant improvement
- Different input patterns should produce different outputs (not all collapsed to one value)
- Gradients should flow properly during training

## Design Considerations

### Input Range Assumptions

| Node Type | Expected Input Range | Threshold |
|-----------|---------------------|-----------|
| DWN | [0, 1] | 0.5 |
| Hybrid | [0, 1] | 0.5 |
| Probabilistic | [0, 1] | (no threshold) |
| UnboundProbabilistic | any | (sigmoid applied) |

### Encoder Compatibility

| Encoder Type | Output Range | Compatible with DWN/Hybrid? |
|--------------|--------------|----------------------------|
| Thermometer | [0, 1] | ✓ Yes (after fix) |
| Binary | {0, 1} | ✓ Yes (0.5 threshold works for both) |
| Gray | {0, 1} | ✓ Yes |

### Why 0.5?

The threshold of 0.5 works for both:
- **Binary inputs** `{0, 1}`: 0 < 0.5 → 0, 1 > 0.5 → 1 ✓
- **Continuous inputs** `[0, 1]`: Natural midpoint threshold ✓
- **Thermometer encoding**: Values below threshold → 0, above → 1 ✓

## Alternative Solutions Considered

1. **Modify encoder to output {0, 1}**: Would break Probabilistic node
2. **Use separate encoders**: Duplicates code and complexity
3. **Adaptive threshold**: Over-engineering for this use case
4. **✓ Use 0.5 threshold**: Simple, works for all cases

## Related Files

- `difflut/nodes/dwn_node.py`
- `difflut/nodes/hybrid_node.py`
- `difflut/nodes/cuda/efd_cuda_kernel.cu`
- `difflut/nodes/cuda/hybrid_cuda_kernel.cu`
- `tests/test_threshold_fix.py` (new test)

## Impact

This is a **critical bug fix** that:
- Fixes DWN and Hybrid nodes to work with standard encoders
- Makes them compatible with the rest of the architecture
- Should significantly improve experimental results
- No negative impact on other nodes

---

**Status:** Ready for recompilation and testing
**Priority:** HIGH - Critical for DWN/Hybrid functionality
**Breaking Change:** No (only fixes existing bug)
