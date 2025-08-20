#!/bin/bash

# TriadFuse Installation Script
# This script installs all dependencies and verifies the installation

set -e  # Exit on error

echo "==================================="
echo "   TriadFuse Installation Script   "
echo "==================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi
echo "✓ Python $python_version"
echo ""

# Check CUDA availability (optional)
echo "Checking CUDA availability..."
if python3 -c "import torch; print('CUDA available' if torch.cuda.is_available() else 'CUDA not available')" 2>/dev/null | grep -q "CUDA available"; then
    echo "✓ CUDA is available"
    cuda_available=true
else
    echo "⚠ CUDA not available - will use CPU (slower)"
    cuda_available=false
fi
echo ""

# Install package in editable mode
echo "Installing TriadFuse package..."
pip install -e . --quiet --no-cache-dir
echo "✓ Package installed"
echo ""

# Install additional dependencies based on platform
if [[ "$OSTYPE" == "linux-gnu"* ]] && [ "$cuda_available" = true ]; then
    echo "Installing Linux-specific dependencies (bitsandbytes)..."
    pip install bitsandbytes --quiet --no-cache-dir
    echo "✓ bitsandbytes installed"
    echo ""
fi

# Verify imports
echo "Verifying installation..."
echo ""

python3 -c "
import sys
print('Testing imports...')

try:
    import torch
    print('✓ PyTorch:', torch.__version__)
except ImportError as e:
    print('❌ PyTorch import failed:', e)
    sys.exit(1)

try:
    import torchvision
    print('✓ TorchVision')
except ImportError as e:
    print('❌ TorchVision import failed:', e)
    sys.exit(1)

try:
    import transformers
    print('✓ Transformers:', transformers.__version__)
except ImportError as e:
    print('❌ Transformers import failed:', e)
    sys.exit(1)

try:
    import kornia
    print('✓ Kornia')
except ImportError as e:
    print('❌ Kornia import failed:', e)
    sys.exit(1)

try:
    import albumentations
    print('✓ Albumentations')
except ImportError as e:
    print('❌ Albumentations import failed:', e)
    sys.exit(1)

try:
    import lpips
    print('✓ LPIPS')
except ImportError as e:
    print('❌ LPIPS import failed:', e)
    sys.exit(1)

try:
    from triadfuse import EOT, TextureHead, DonutSurrogate
    print('✓ TriadFuse modules')
except ImportError as e:
    print('❌ TriadFuse import failed:', e)
    sys.exit(1)

print('')
print('All imports successful!')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Installation verification failed"
    exit 1
fi

echo ""
echo "==================================="
echo "   Installation Complete! 🎉       "
echo "==================================="
echo ""
echo "Quick test:"
echo "  python scripts/attack_train.py --help"
echo ""
echo "Run tests:"
echo "  pytest tests/ -v"
echo ""
echo "See README.md for usage examples."
