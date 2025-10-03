#!/bin/bash
# Python wrapper script for VSCode to use Singularity container
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${SCRIPT_DIR}/difflogic.sif"

exec singularity exec --nv "$CONTAINER" python "$@"
