#!/bin/bash
set -e

# Build script for difflogic Apptainer container
# This script builds the container with DiffLUT CUDA extensions compiled

echo "=============================================="
echo "Building Difflogic Apptainer Container"
echo "=============================================="

# Get script directory (project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Container configuration
CONTAINER_NAME="difflut.sif"
DEF_FILE="${PROJECT_ROOT}/singularity/difflut.def"
OUTPUT_FILE="${PROJECT_ROOT}/singularity/${CONTAINER_NAME}"

# Check if definition file exists
if [ ! -f "$DEF_FILE" ]; then
    echo "Error: Container definition file not found: $DEF_FILE"
    exit 1
fi

# Check if difflut directory exists
if [ ! -d "${PROJECT_ROOT}/difflut" ]; then
    echo "Error: difflut directory not found at ${PROJECT_ROOT}/difflut"
    echo "This is required to build the container with CUDA extensions"
    exit 1
fi

# Warn if container already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Warning: Container already exists at $OUTPUT_FILE"
    read -p "Do you want to rebuild? This will delete the existing container. (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Build cancelled"
        exit 0
    fi
    echo "Removing existing container..."
    rm -f "$OUTPUT_FILE"
fi

# Build the container
echo "Building container from: $DEF_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Building from project root: $PROJECT_ROOT"
echo ""
echo "Note: This will compile DiffLUT CUDA extensions for GPU architectures:"
echo "  - 6.1 (Titan XP)"
echo "  - 7.0 (V100)"
echo "  - 7.5 (Titan RTX, RTX 2080 Ti)"
echo "  - 8.0 (A100)"
echo "  - 8.6 (RTX 3090, A6000)"
echo ""

cd "$PROJECT_ROOT"

# Set cache directory for faster image extraction
# export APPTAINER_CACHEDIR="${HOME}/.apptainer/cache"
# export SINGULARITY_CACHEDIR="${HOME}/.singularity/cache"
# mkdir -p "$APPTAINER_CACHEDIR" "$SINGULARITY_CACHEDIR"

# echo "Using cache directory: $APPTAINER_CACHEDIR"
# echo ""

# Build with apptainer (requires root/fakeroot or --remote)
if command -v apptainer &> /dev/null; then
    echo "Pre-pulling base images to cache (this speeds up subsequent builds)..."
    apptainer cache list 2>/dev/null || true
    echo ""
    apptainer build "$OUTPUT_FILE" "$DEF_FILE"
elif command -v singularity &> /dev/null; then
    echo "Pre-pulling base images to cache (this speeds up subsequent builds)..."
    singularity cache list 2>/dev/null || true
    echo ""
    singularity build "$OUTPUT_FILE" "$DEF_FILE"
else
    echo "Error: Neither apptainer nor singularity command found"
    echo "Please install Apptainer/Singularity to build containers"
    exit 1
fi

# Check if build was successful
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "=============================================="
    echo "Build completed successfully!"
    echo "=============================================="
    echo "Container location: $OUTPUT_FILE"
    echo "Container size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo ""
    echo "To test the container:"
    echo "  apptainer exec --nv $OUTPUT_FILE python3 -c 'import difflut; print(difflut.__version__)'"
    echo ""
    echo "To use the container for training:"
    echo "  ./train.sh <hydra_args>"
else
    echo ""
    echo "=============================================="
    echo "Build failed!"
    echo "=============================================="
    exit 1
fi
