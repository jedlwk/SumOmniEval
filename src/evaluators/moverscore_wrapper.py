"""
MoverScore wrapper that ensures CPU-only mode.

This module MUST be imported instead of importing moverscore_v2 directly.
It sets up the environment correctly before moverscore initializes.
"""

import os

# CRITICAL: Set CPU-only mode BEFORE any PyTorch/MoverScore imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['USE_CUDA'] = '0'
os.environ['FORCE_CPU'] = '1'

# Import torch first and verify CPU mode
try:
    import torch
    if torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is available in PyTorch. MoverScore requires CPU-only PyTorch. "
            "Please reinstall PyTorch: pip3 uninstall torch && "
            "pip3 install torch --index-url https://download.pytorch.org/whl/cpu"
        )
except ImportError:
    raise ImportError("PyTorch not installed. Run ./setup.sh to install dependencies.")

# Now safe to import moverscore
try:
    from moverscore_v2 import word_mover_score, get_idf_dict

    # Export the functions
    __all__ = ['word_mover_score', 'get_idf_dict']

except (ImportError, AssertionError, RuntimeError) as e:
    error_msg = str(e)
    if "CUDA" in error_msg or "cuda" in error_msg.lower():
        raise RuntimeError(
            f"MoverScore CUDA initialization error: {error_msg}\n\n"
            "This means PyTorch is trying to use CUDA. Fix by running:\n"
            "  ./setup.sh\n"
            "This will reinstall PyTorch (CPU-only) and all dependencies."
        )
    else:
        raise ImportError(f"Could not import MoverScore: {error_msg}")
