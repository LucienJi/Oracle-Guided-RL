#!/bin/bash
# Oracle-Guided-RL quick install script
# Usage: bash install.sh

set -euo pipefail

echo "=========================================="
echo "Oracle-Guided-RL environment installation script"
echo "=========================================="

# Check whether conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    echo "Download: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Resolve the script directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

require_dir() {
    local path="$1"
    if [[ ! -d "$path" ]]; then
        echo "Error: missing dependency directory $path"
        echo "This repository cannot be reproduced from a clean clone without these third_party dependencies."
        echo "Minimal fix: convert these dependencies to git submodules or provide a bootstrap script pinned to exact commits."
        exit 1
    fi
}

echo ""
echo "Step 1/4: Create the Conda environment..."
echo "----------------------------------------"
conda env create -f environment.yml


echo ""
echo "Step 2/4: Validate the installation..."
echo "----------------------------------------"

# Validate key dependencies
conda run -n oracles python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || echo "✗ PyTorch is not installed correctly"
conda run -n oracles python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" || echo "✗ CUDA check failed"
conda run -n oracles python -c "import gymnasium; print(f'✓ Gymnasium: {gymnasium.__version__}')" || echo "✗ Gymnasium is not installed correctly"
conda run -n oracles python -c "import dm_control; print('✓ dm-control installed')" || echo "✗ dm-control is not installed correctly"

echo ""
echo "Step 3/4: Environment variable notes..."
echo "----------------------------------------"
echo "Did not modify ~/.bashrc."
echo "If you need headless rendering, set this explicitly in the current shell or job script:"
echo "  export MUJOCO_GL=egl"

echo ""
echo "Step 4/4: Installation complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  1. Activate the environment: conda activate oracles"
echo "  2. Go to the project directory: cd $SCRIPT_DIR"
echo "  3. See README.md for the canonical setup / quick start path"
echo ""
echo "If you run into issues, see INSTALL.md for detailed instructions."
echo ""
