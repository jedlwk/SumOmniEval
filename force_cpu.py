"""
Force CPU-only mode for all PyTorch operations.
This must be imported BEFORE any other imports in the application.
"""
import os
import sys

# Force CPU mode - set BEFORE any imports (multiple variations for maximum compatibility)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Disable CUDA in PyTorch (if not yet imported)
os.environ['USE_CUDA'] = '0'
os.environ['FORCE_CPU'] = '1'

# Verify PyTorch will use CPU (if already installed)
try:
    import torch
    if torch.cuda.is_available():
        print("⚠️  WARNING: CUDA is still available after forcing CPU mode!", file=sys.stderr)
        print("⚠️  This may cause issues with MoverScore.", file=sys.stderr)
    else:
        print("✅ CPU-only mode enforced successfully", file=sys.stderr)
except ImportError:
    # PyTorch not yet installed - this is fine during initial setup
    print("✅ CPU environment variables set (PyTorch not yet loaded)", file=sys.stderr)
