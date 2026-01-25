#!/usr/bin/env python3
"""
MoverScore CPU Compatibility Patch

This utility patches the MoverScore library to enable CPU-only operation.
The original MoverScore package hardcodes CUDA device usage, causing failures
on systems without GPU support. This script modifies the installed package to:

1. Automatically detect CUDA availability and fall back to CPU
2. Fix NumPy compatibility issues (deprecated np.float → np.float64)

Usage:
    python3 patch_moverscore.py

Note: Run this after installing moverscore_v2 if you encounter CUDA errors.
"""

import os
import sys

def find_moverscore_file():
    """Find the installed moverscore_v2.py file."""
    try:
        import moverscore_v2
        # This will fail during import, but we can catch it
    except:
        pass

    # Try common installation locations
    import site
    locations = [
        site.getusersitepackages(),
        *site.getsitepackages(),
    ]

    for location in locations:
        if location:
            file_path = os.path.join(location, 'moverscore_v2.py')
            if os.path.exists(file_path):
                return file_path

    return None

def patch_moverscore(file_path):
    """Patch moverscore_v2.py to support CPU mode and NumPy compatibility."""
    import re

    print(f"Patching {file_path}...")

    with open(file_path, 'r') as f:
        content = f.read()

    patches_applied = []

    # Patch 1: Replace hardcoded cuda device with conditional check
    original_cuda = "device = 'cuda'"
    patched_cuda = "device = 'cuda' if torch.cuda.is_available() else 'cpu'"

    if original_cuda in content and "torch.cuda.is_available()" not in content:
        content = content.replace(original_cuda, patched_cuda)
        patches_applied.append("CUDA device check")
        print(f"✅ Patch 1: CUDA device")
        print(f"   Changed: {original_cuda}")
        print(f"   To:      {patched_cuda}")
    elif "torch.cuda.is_available()" in content:
        print("✅ Patch 1: CUDA device (already applied)")
    else:
        print(f"⚠️  Patch 1: Could not find '{original_cuda}'")

    # Patch 2: Replace deprecated np.float with np.float64
    # Use regex to only replace np.float when NOT followed by a digit (to preserve np.float32, np.float64, etc.)
    pattern = r'dtype=np\.float(?!\d)'
    replacement = 'dtype=np.float64'

    matches = re.findall(pattern, content)
    if matches:
        content = re.sub(pattern, replacement, content)
        patches_applied.append("NumPy float compatibility")
        print(f"✅ Patch 2: NumPy compatibility")
        print(f"   Replaced {len(matches)} occurrences of dtype=np.float with dtype=np.float64")
    else:
        print("✅ Patch 2: NumPy compatibility (already applied)")

    # Write back if any patches were applied
    if patches_applied:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"\n✅ Successfully patched MoverScore ({', '.join(patches_applied)})")
        return True
    else:
        print("✅ MoverScore already fully patched")
        return True

def verify_patch():
    """Verify that MoverScore can be imported after patching."""
    print("\nVerifying patched MoverScore...")
    try:
        # Force CPU mode
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        import moverscore_v2
        print("✅ MoverScore imports successfully!")
        return True
    except Exception as e:
        print(f"❌ MoverScore still fails to import: {e}")
        return False

if __name__ == '__main__':
    print("="*80)
    print("MoverScore CPU Patch")
    print("="*80)
    print()

    # Find moverscore file
    file_path = find_moverscore_file()

    if not file_path:
        print("❌ Could not find moverscore_v2.py")
        print("   Make sure MoverScore is installed first.")
        sys.exit(1)

    print(f"Found MoverScore at: {file_path}")
    print()

    # Patch the file
    if not patch_moverscore(file_path):
        sys.exit(1)

    print()

    # Verify it works
    if verify_patch():
        print()
        print("="*80)
        print("✅ MoverScore is now ready for CPU mode!")
        print("="*80)
        sys.exit(0)
    else:
        sys.exit(1)
