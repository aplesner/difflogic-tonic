#!/bin/bash
# Script to recompile CUDA extensions after threshold fix

echo "=========================================="
echo "Recompiling CUDA extensions"
echo "Threshold fix: 0.0 -> 0.5 for [0,1] inputs"
echo "=========================================="

cd /itet-stor/sbuehrer/net_scratch/difflut

# Remove old compiled extensions
echo "Removing old compiled extensions..."
rm -f efd_cuda*.so fourier_cuda*.so hybrid_cuda*.so
find . -name "*.so" -path "*/difflut/nodes/cuda/*" -delete

# Clean build artifacts
echo "Cleaning build artifacts..."
rm -rf build/ *.egg-info/

# Reinstall the package (this will recompile CUDA extensions)
echo "Recompiling CUDA extensions..."
pip install -e . --force-reinstall --no-deps

echo ""
echo "=========================================="
echo "CUDA extensions recompiled successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run test: python tests/test_threshold_fix.py"
echo "2. Re-run experiments to see improved performance"
echo ""
